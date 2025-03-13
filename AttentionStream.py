import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter
import os
from rouge_score import rouge_scorer
import torch.nn.functional as F
import math

# Load and preprocess data into question-answer pairs
def load_qa_pairs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n\n')  # Split by empty lines
    qa_pairs = []
    for block in lines:
        lines_in_block = block.strip().split('\n')
        question = lines_in_block[0].strip()
        answer = ' '.join(line.strip() for line in lines_in_block[1:]) if len(lines_in_block) > 1 else ''
        qa_pairs.append((question, answer))
    return qa_pairs

qa_pairs = load_qa_pairs('data.txt')

# Tokenization and vocabulary
all_text = ' '.join(q + ' ' + a for q, a in qa_pairs)
words = all_text.split()
word_counts = Counter(words)
word_to_index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
word_to_index.update({word: i+3 for i, (word, _) in enumerate(word_counts.items())})
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(word_to_index)

# Encode QA pairs
def encode_sequence(seq, max_len):
    tokens = seq.split()
    encoded = [word_to_index.get(word, 0) for word in tokens]
    encoded = encoded[:max_len] + [word_to_index['<EOS>']]
    return encoded + [word_to_index['<PAD>']] * (max_len + 1 - len(encoded))

max_seq_len = 20  # Adjust based on your data
X = [encode_sequence(q, max_seq_len) for q, _ in qa_pairs]
y = [encode_sequence(a, max_seq_len) for _, a in qa_pairs]

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Dataset and DataLoader
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# Model components
class SequenceMergingSeq(nn.Module):
    def __init__(self, embedding_dim):
        super(SequenceMergingSeq, self).__init__()
        self.embedding_dim = embedding_dim
        self.time_modulation = nn.Parameter(torch.ones(1, 1, embedding_dim) * 0.5)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, C, V, W):
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

    def forward(self, x, x_prev):
        shifted = self.shift(x, x_prev)
        merged = self.sequence_merging(shifted, shifted, shifted)
        coupled = self.state_coupling(merged)
        return shifted + coupled

class AttentionStreamEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=2):
        super(AttentionStreamEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList([
            AttentionStreamBlock(embedding_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None)
        return x.mean(dim=1)  # (batch_size, embedding_dim)

class AttentionStreamDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=2):
        super(AttentionStreamDecoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList([
            AttentionStreamBlock(embedding_dim) for _ in range(num_layers)
        ])
        self.cross_attention = nn.Linear(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, encoder_output):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None)
        attn_weights = self.cross_attention(encoder_output).unsqueeze(1)
        x = x + attn_weights
        return self.fc(x[:, -1, :])  # Predict next token

class AttentionStreamSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=2):
        super(AttentionStreamSeq2Seq, self).__init__()
        self.encoder = AttentionStreamEncoder(vocab_size, embedding_dim, num_layers)
        self.decoder = AttentionStreamDecoder(vocab_size, embedding_dim, num_layers)
        self.embedding_dim = embedding_dim

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        encoder_output = self.encoder(input_seq)  # (batch_size, embedding_dim)
        batch_size = input_seq.size(0)
        
        # Decoder input starts with <SOS>
        decoder_input = torch.full((batch_size, 1), word_to_index['<SOS>'], device=input_seq.device)
        outputs = []
        
        for t in range(max_seq_len + 1):  # +1 for <EOS>
            output = self.decoder(decoder_input, encoder_output)
            outputs.append(output)
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio and t < target_seq.size(1)
            top1 = output.argmax(-1).unsqueeze(1)
            decoder_input = target_seq[:, t].unsqueeze(1) if use_teacher_forcing else top1
        
        return torch.stack(outputs, dim=1)  # (batch_size, max_seq_len+1, vocab_size)

# Hyperparameters
embedding_dim = 128  # Increased for better representation
num_layers = 2
model = AttentionStreamSeq2Seq(vocab_size, embedding_dim, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=word_to_index['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
model_path = 'attention_stream_seq2seq.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded saved model.")
else:
    epochs = 100  # Increased epochs due to small dataset
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X, batch_y, teacher_forcing_ratio=0.5)
            loss = criterion(outputs.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

# Perplexity calculation
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

perplexity = calculate_perplexity(model, eval_dataloader, criterion, device)
print(f'Perplexity on validation set: {perplexity:.4f}')

# Inference
def generate_response(input_seq, model, word_to_index, index_to_word, max_len=20):
    model.eval()
    input_seq = encode_sequence(input_seq, max_seq_len)
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_output = model.encoder(input_tensor)
        decoder_input = torch.tensor([[word_to_index['<SOS>']]], device=device)
        generated = []
        
        for _ in range(max_len):
            output = model.decoder(decoder_input, encoder_output)
            next_word_idx = output.argmax(-1).item()
            if next_word_idx == word_to_index['<EOS>']:
                break
            generated.append(index_to_word[next_word_idx])
            decoder_input = torch.tensor([[next_word_idx]], device=device)
    
    return ' '.join(generated)

# Interactive loop
print("Model ready! Type a question (exit to quit):")
while True:
    input_seq = input("Input: ")
    if input_seq.lower() == "exit":
        break
    response = generate_response(input_seq, model, word_to_index, index_to_word)
    print(f"Generated: {response}\n")