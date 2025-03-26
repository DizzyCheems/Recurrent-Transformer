import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter
import os
import math
import time
import asyncio
from backend import refine_response  # Import the backend function

# Define ANSI color codes
BLUE = '\033[94m'
GREEN = '\033[92m'
RESET = '\033[0m'

# Load and preprocess data (case-insensitive without changing dataset format)
def load_qa_pairs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n\n')
    qa_pairs = []
    for block in lines:
        lines_in_block = block.strip().split('\n')
        question = lines_in_block[0].strip()
        answer = ' '.join(line.strip() for line in lines_in_block[1:]) if len(lines_in_block) > 1 else ''
        qa_pairs.append((question, answer))
    return qa_pairs

qa_pairs = load_qa_pairs('data.txt')
all_text = ' '.join((q + ' ' + a).lower() for q, a in qa_pairs)
words = all_text.split()
word_counts = Counter(words)
word_to_index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
word_to_index.update({word: i+3 for i, (word, _) in enumerate(word_counts.items())})
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(word_to_index)

def encode_sequence(seq, max_len):
    tokens = seq.lower().split()
    encoded = [word_to_index.get(word, 0) for word in tokens]
    encoded = encoded[:max_len] + [word_to_index['<EOS>']]
    return encoded + [word_to_index['<PAD>']] * (max_len + 1 - len(encoded))

max_seq_len = 20
X = [encode_sequence(q, max_seq_len) for q, _ in qa_pairs]
y = [encode_sequence(a, max_seq_len) for _, a in qa_pairs]
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
train_dataloader = DataLoader(train_dataset, batch_size=180, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=180, shuffle=False)

# Model Definition
class SequenceMergingSeq(nn.Module):
    def __init__(self, embedding_dim):
        super(SequenceMergingSeq, self).__init__()
        self.embedding_dim = embedding_dim
        self.time_modulation = nn.Parameter(torch.ones(1, 1, embedding_dim) * 0.5)
        self.context_modulation = nn.Parameter(torch.ones(1, embedding_dim) * 0.5)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, C, V, W, encoder_context=None):
        batch_size, seq_len, _ = C.shape
        a = torch.zeros(batch_size, 1, device=C.device)
        b = torch.zeros(batch_size, self.embedding_dim, device=C.device)
        outputs = []
        for t in range(seq_len):
            C_t = C[:, t, :]
            V_t = V[:, t, :]
            W_t = W[:, t, :]
            decay = torch.sigmoid(W_t * self.time_modulation.expand(batch_size, -1, -1).squeeze(1))
            a = decay.mean(dim=1, keepdim=True) * a + torch.exp(C_t).sum(dim=1, keepdim=True)
            b = decay * b + (torch.exp(C_t) * V_t)
            if encoder_context is not None:
                context_weight = torch.sigmoid(encoder_context * self.context_modulation.expand(batch_size, -1))
                b = b + (context_weight * encoder_context)
            output_t = b / (a + 1e-8)
            outputs.append(output_t.unsqueeze(1))
        result = torch.cat(outputs, dim=1)
        return self.layer_norm(result)

class StateCoupling(nn.Module):
    def __init__(self, input_dim):
        super(StateCoupling, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        gate = self.sigmoid(self.Wr(x))
        value = self.Wv(x)
        return self.layer_norm(gate * value)

class Shift(nn.Module):
    def __init__(self, input_dim):
        super(Shift, self).__init__()
        self.mu = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, x_prev):
        if x_prev is None:
            return x
        return self.mu * x + (1 - self.mu) * x_prev

class AttentionStreamBlock(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionStreamBlock, self).__init__()
        self.shift = Shift(embedding_dim)
        self.sequence_merging = SequenceMergingSeq(embedding_dim)
        self.state_coupling = StateCoupling(embedding_dim)

    def forward(self, x, x_prev, encoder_context=None):
        shifted = self.shift(x, x_prev)
        merged = self.sequence_merging(shifted, shifted, shifted, encoder_context)
        coupled = self.state_coupling(merged)
        return shifted + coupled

class AttentionStreamEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=2):
        super(AttentionStreamEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList([AttentionStreamBlock(embedding_dim) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None)
        return x.mean(dim=1)

class AttentionStreamDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=2):
        super(AttentionStreamDecoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList([AttentionStreamBlock(embedding_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, encoder_output):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None, encoder_output)
        return self.fc(x[:, -1, :])

class AttentionStreamSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=2):
        super(AttentionStreamSeq2Seq, self).__init__()
        self.encoder = AttentionStreamEncoder(vocab_size, embedding_dim, num_layers)
        self.decoder = AttentionStreamDecoder(vocab_size, embedding_dim, num_layers)
        self.embedding_dim = embedding_dim

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        encoder_output = self.encoder(input_seq)
        batch_size = input_seq.size(0)
        decoder_input = torch.full((batch_size, 1), word_to_index['<SOS>'], device=input_seq.device)
        outputs = []
        for t in range(max_seq_len + 1):
            output = self.decoder(decoder_input, encoder_output)
            outputs.append(output)
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio and t < target_seq.size(1)
            top1 = output.argmax(-1).unsqueeze(1)
            decoder_input = target_seq[:, t].unsqueeze(1) if use_teacher_forcing else top1
        return torch.stack(outputs, dim=1)

# Training and Evaluation
def train_model(model, model_name, dataloader, criterion, optimizer, epochs, device):
    model_path = f'{model_name}.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded saved {model_name}.")
        return 0
    else:
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X, batch_y)
                loss = criterion(outputs.view(-1, vocab_size), batch_y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f'{model_name} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), model_path)
        print(f"{model_name} saved.")
        return time.time() - start_time

def calculate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X, batch_y, teacher_forcing_ratio=0.0)
            loss = criterion(outputs.view(-1, vocab_size), batch_y.view(-1))
            total_loss += loss.item() * batch_X.size(0)
            total_words += (batch_y != word_to_index['<PAD>']).sum().item()
    avg_loss = total_loss / total_words
    return math.exp(avg_loss) if total_words > 0 else float('nan')

# Optimized generate_response
async def generate_response(model, input_seq, word_to_index, index_to_word, max_len=20, temperature=0.7, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        # GPU: AttentionStream generation
        encoded_input = torch.tensor(encode_sequence(input_seq.lower(), max_seq_len), dtype=torch.long, device=device).unsqueeze(0)
        encoder_output = model.encoder(encoded_input)
        decoder_input = torch.full((1, 1), word_to_index['<SOS>'], device=device)
        generated = []
        generated_ids = set()
        
        for _ in range(max_len):
            output = model.decoder(decoder_input, encoder_output)
            output = output / temperature
            for idx in generated_ids:
                output[0, idx] /= repetition_penalty
            probs = torch.softmax(output, dim=-1)
            next_word_idx = torch.multinomial(probs, 1).item()
            if next_word_idx == word_to_index['<EOS>']:
                break
            generated.append(index_to_word[next_word_idx])
            generated_ids.add(next_word_idx)
            decoder_input = torch.tensor([[next_word_idx]], dtype=torch.long, device=device)
        
        attentionstream_response = ' '.join(generated)
        
        # Call backend for refinement
        refined_response = await refine_response(input_seq, attentionstream_response)
        return refined_response

# Setup and Train
embedding_dim = 128
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionStreamSeq2Seq(vocab_size, embedding_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_index['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_time = train_model(model, "attention_stream_seq2seq_time_mod", train_dataloader, criterion, optimizer, 180, device)
perplexity = calculate_perplexity(model, eval_dataloader, criterion, device)
print(f"Perplexity: {perplexity:.4f}, Training Time: {training_time:.2f}s")

# Interactive Testing with colored output
print("Model ready! Type a question (exit to quit):")
async def main_loop():
    while True:
        input_seq = input(f"{BLUE}Input: {RESET}")
        if input_seq.lower() == "exit":
            break
        response = await generate_response(model, input_seq, word_to_index, index_to_word)
        print(f"{GREEN}Generated: {response}{RESET}\n")

if __name__ == "__main__":
    asyncio.run(main_loop())