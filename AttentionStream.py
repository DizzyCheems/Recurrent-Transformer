import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter
import os
from rouge_score import rouge_scorer
import torch.nn.functional as F

# Load the data
with open('data.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Tokenization and creating a word-to-index mapping
words = data.split()
word_counts = Counter(words)
word_to_index = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
index_to_word = {i: word for word, i in word_to_index.items()}

# Encode words as indices
encoded_data = [word_to_index[word] for word in words]

# Prepare input-output pairs (context, next word)
seq_length = 5
X = []
y = []
for i in range(len(encoded_data) - seq_length):
    X.append(encoded_data[i:i+seq_length])
    y.append(encoded_data[i+seq_length])

X = np.array(X)
y = np.array(y)

# Convert to tensors
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Dataset processor
dataset = TensorDataset(X, y)

# Split into 80% training and 20% evaluation
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

class SequenceMergingSeq(nn.Module):
    def __init__(self, embedding_dim):
        super(SequenceMergingSeq, self).__init__()
        self.embedding_dim = embedding_dim
        # learnable Time Modulation parameter
        self.time_modulation = nn.Parameter(torch.ones(1, 1, embedding_dim) * 0.5)  # Learnable decay per dimension
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, C, V, W):
        batch_size, seq_len, _ = C.shape
        
        a = torch.zeros(batch_size, 1, device=C.device)
        b = torch.zeros(batch_size, self.embedding_dim, device=C.device)
        
        outputs = []
        for t in range(seq_len):
            C_t = C[:, t, :]  # Current token
            V_t = V[:, t, :]
            W_t = W[:, t, :]
            
            # Time Modulation: Dynasmic decay based on W_t and learnable parameter
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
        return shifted + coupled  # Residual connection

class AttentionStreamModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=1):
        super(AttentionStreamModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList([ 
            AttentionStreamBlock(embedding_dim)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None)
        return self.fc(x[:, -1, :])  # Last token output

# Hyperparameters
embedding_dim = 50
vocab_size = len(word_to_index) + 1
model = AttentionStreamModel(vocab_size, embedding_dim)

# Host Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Save the Model
model_path = 'attention_stream_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded saved model.")
else:
    # Training iterations
    epochs = 180
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_dataloader)}')
    
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

# ROUGE Scorer Evaluators
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge_score(reference, generated):
    reference = ' '.join(reference)
    generated = ' '.join(generated)
    scores = scorer.score(reference, generated)
    return scores

# Softmax with temperature
def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

def predict_next_word(sequence, word_to_index, index_to_word, model, temperature=1.0):
    model.eval()
    sequence = [word_to_index.get(word, 0) for word in sequence.split()]
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sequence)
        next_word_idx = sample_with_temperature(output, temperature)
        next_word = index_to_word[next_word_idx.item()]
        return next_word

# Define explicit reference sequences for certain inputs
reference_sequences = {
    "count 1 to 10": [
        "Here", "are", "the", "first", "10", "natural", "numbers", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
    ],
    "What is the fastest land animal?": [
        "The", "fastest", "land", "animal", "is", "the", "cheetah", "which", "can", "reach", "speeds", "of", "up", "to", "60", "miles", "per", "hour", "97", "kilometers", "per", "hour", "in", "short", "bursts", "covering", "distances", "of", "up", "to", "1500", "feet", "460", "meters"
    ]
}

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2, dim=-1)

def get_sequence_embedding(sequence, model, word_to_index, device):
    sequence_indices = [word_to_index.get(word, 0) for word in sequence]
    sequence_tensor = torch.tensor(sequence_indices, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = model.embeddings(sequence_tensor)
        return embeddings.mean(dim=1)

# Interactive generation with ROUGE score calculation
print("\033[94mModel ready! Type a sequence (exit to quit):\033[0m")
while True:
    input_seq = input("\033[94mInput: \033[0m")
    if input_seq.lower() == "exit":
        break
    
    generated = input_seq.split()
    reference = reference_sequences.get(input_seq, input_seq.split())

    for _ in range(20):
        context = ' '.join(generated[-seq_length:])
        next_word = predict_next_word(context, word_to_index, index_to_word, model, temperature=0.8)
        generated.append(next_word)
    
    rouge_score = compute_rouge_score(reference, generated[len(input_seq.split()):])
    print(f"ROUGE Score: {rouge_score}")
    
    reference_embedding = get_sequence_embedding(reference, model, word_to_index, device)
    generated_embedding = get_sequence_embedding(generated[len(input_seq.split()):], model, word_to_index, device)
    similarity = cosine_similarity(reference_embedding, generated_embedding)
    print(f"Cosine Similarity: {similarity.item():.4f}")
    
    print(f"\033[92mGenerated: {' '.join(generated[len(input_seq.split()):])}\033[0m\n")