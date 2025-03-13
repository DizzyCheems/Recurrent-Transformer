import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import Counter
import numpy as np
import os
from rouge_score import rouge_scorer
import math
import matplotlib.pyplot as plt

# Load and preprocess data
with open('data.txt', 'r', encoding='utf-8') as file:
    data = file.read()

words = data.split()
word_counts = Counter(words)
word_to_index = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}  # Start at 1 (0 for padding)
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(word_to_index) + 1

encoded_data = [word_to_index[word] for word in words]
seq_length = 5
X, y = [], []
for i in range(len(encoded_data) - seq_length):
    X.append(encoded_data[i:i+seq_length])
    y.append(encoded_data[i+seq_length])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hybrid Model (Attention-Stream Model)
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
        for t in range(seq_len):  # Fixed to use seq_len instead of seq_length
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

class AttentionStreamModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers=1):
        super(AttentionStreamModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList([AttentionStreamBlock(embedding_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None)
        return self.fc(x[:, -1, :])

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads, n_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.embedding_dim = embedding_dim

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer_encoder(x, mask)
        return self.fc(x[:, -1, :])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Training Function
def train_model(model, train_dataloader, eval_dataloader, epochs, device, model_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded saved {model.__class__.__name__} model.")
    else:
        model.train()
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
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_dataloader):.4f}')
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model.__class__.__name__} model.")

# Evaluation Functions
def calculate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            total_words += batch_X.size(0)
    avg_loss = total_loss / total_words
    return math.exp(avg_loss)

def predict_sequence(model, input_seq, word_to_index, index_to_word, seq_length, device, max_len=20, temperature=0.8):
    model.eval()
    seq = [word_to_index.get(word, 0) for word in input_seq.split()]
    if len(seq) < seq_length:
        seq = [0] * (seq_length - len(seq)) + seq
    seq = seq[-seq_length:]
    generated = seq.copy()
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([generated[-seq_length:]], dtype=torch.long).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output / temperature, dim=-1)
            next_word_idx = torch.multinomial(probs, 1).item()
            generated.append(next_word_idx)
    return [index_to_word.get(idx, "<UNK>") for idx in generated[seq_length:]]

def evaluate_model(model, dataloader, word_to_index, index_to_word, seq_length, device):
    criterion = nn.CrossEntropyLoss()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    perplexity = calculate_perplexity(model, dataloader, criterion, device)
    rouge_scores = []
    cosine_similarities = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(dataloader):
            if i >= 10:  # Limit to 10 samples for efficiency
                break
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            input_seq = ' '.join([index_to_word[idx.item()] for idx in batch_X[0]])
            reference = [index_to_word[idx.item()] for idx in batch_y[:seq_length]]
            generated = predict_sequence(model, input_seq, word_to_index, index_to_word, seq_length, device)
            
            # ROUGE
            ref_text = ' '.join(reference)
            gen_text = ' '.join(generated)
            scores = scorer.score(ref_text, gen_text)
            rouge_scores.append(scores)
            
            # Cosine Similarity
            # Use the correct embedding attribute based on model type
            embedding_layer = model.embeddings if isinstance(model, AttentionStreamModel) else model.embedding
            ref_emb = embedding_layer(batch_y[:seq_length].unsqueeze(0).to(device)).mean(dim=1)
            gen_ids = torch.tensor([[word_to_index.get(w, 0) for w in generated]], dtype=torch.long).to(device)
            gen_emb = embedding_layer(gen_ids).mean(dim=1)
            cos_sim = F.cosine_similarity(ref_emb, gen_emb).item()
            cosine_similarities.append(cos_sim)
    
    avg_rouge = {
        "rouge1": np.mean([s["rouge1"].fmeasure for s in rouge_scores]),
        "rouge2": np.mean([s["rouge2"].fmeasure for s in rouge_scores]),
        "rougeL": np.mean([s["rougeL"].fmeasure for s in rouge_scores])
    }
    avg_cos_sim = np.mean(cosine_similarities)
    
    return {"perplexity": perplexity, "rouge": avg_rouge, "cosine_similarity": avg_cos_sim}

# Initialize and Train Models
embedding_dim = 64  # Changed to be divisible by n_heads=4
hybrid_model = AttentionStreamModel(vocab_size, embedding_dim, num_layers=1).to(device)
transformer_model = TransformerModel(vocab_size, embedding_dim, n_heads=4, n_layers=2).to(device)

train_model(hybrid_model, train_dataloader, eval_dataloader, epochs=50, device=device, model_path="hybrid_model.pth")
train_model(transformer_model, train_dataloader, eval_dataloader, epochs=50, device=device, model_path="transformer_model.pth")

# Evaluate Models
hybrid_results = evaluate_model(hybrid_model, eval_dataloader, word_to_index, index_to_word, seq_length, device)
transformer_results = evaluate_model(transformer_model, eval_dataloader, word_to_index, index_to_word, seq_length, device)

# Print Results
print("\nHybrid Model Results:")
print(f"Perplexity: {hybrid_results['perplexity']:.4f}")
print(f"ROUGE Scores: {hybrid_results['rouge']}")
print(f"Cosine Similarity: {hybrid_results['cosine_similarity']:.4f}")

print("\nTransformer Model Results:")
print(f"Perplexity: {transformer_results['perplexity']:.4f}")
print(f"ROUGE Scores: {transformer_results['rouge']}")
print(f"Cosine Similarity: {transformer_results['cosine_similarity']:.4f}")

# Visualization with Matplotlib
def plot_comparison(hybrid_results, transformer_results):
    metrics = ['Perplexity', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Cosine Similarity']
    hybrid_values = [
        hybrid_results['perplexity'],
        hybrid_results['rouge']['rouge1'],
        hybrid_results['rouge']['rouge2'],
        hybrid_results['rouge']['rougeL'],
        hybrid_results['cosine_similarity']
    ]
    transformer_values = [
        transformer_results['perplexity'],
        transformer_results['rouge']['rouge1'],
        transformer_results['rouge']['rouge2'],
        transformer_results['rouge']['rougeL'],
        transformer_results['cosine_similarity']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, hybrid_values, width, label='Hybrid Model', color='skyblue')
    ax.bar(x + width/2, transformer_values, width, label='Transformer Model', color='salmon')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Hybrid and Transformer Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Plot the comparison
plot_comparison(hybrid_results, transformer_results)