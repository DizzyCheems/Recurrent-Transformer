import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os
from sklearn.model_selection import train_test_split

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

# Convert to tensors and split data
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save metadata for evaluation
torch.save({
    'word_to_index': word_to_index,
    'index_to_word': index_to_word,
    'X_test': X_test,
    'y_test': y_test,
    'vocab_size': len(word_to_index) + 1
}, 'model_metadata.pth')

# Prepare DataLoader
train_dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class SequenceMergingSeq(nn.Module):
    def __init__(self, embedding_dim):
        super(SequenceMergingSeq, self).__init__()
        self.embedding_dim = embedding_dim
        self.decay_net = nn.Linear(embedding_dim, 1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, C, V, W):
        batch_size, seq_len, _ = C.shape
        
        # Initialize a and b for each sequence in the batch
        a = torch.zeros(batch_size, 1, device=C.device)
        b = torch.zeros(batch_size, self.embedding_dim, device=C.device)
        
        outputs = []
        for t in range(seq_len):
            C_t = C[:, t, :]
            V_t = V[:, t, :]
            W_t = W[:, t, :]
            
            # Update a and b incrementally
            decay = torch.sigmoid(self.decay_net(W_t))
            a = decay * a + torch.exp(C_t).sum(dim=1, keepdim=True)
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
        self.blocks = nn.ModuleList([
            AttentionStreamBlock(embedding_dim)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None)
        return self.fc(x[:, -1, :])

    def generate(self, context, max_length=20, temperature=1.0):
        self.eval()
        generated = []
        current_seq = context.to(self.embeddings.weight.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                output = self(current_seq)
                probs = torch.softmax(output / temperature, dim=1)
                next_idx = torch.argmax(probs, dim=1)
                generated.append(next_idx.item())
                current_seq = torch.cat([current_seq[:, 1:], next_idx.unsqueeze(0)], dim=1)
                
        return generated

# Hyperparameters
embedding_dim = 50
vocab_size = len(word_to_index) + 1
model = AttentionStreamModel(vocab_size, embedding_dim)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Model loading/saving
model_path = 'attention_stream_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded saved model.")
else:
    # Training loop
    epochs = 55
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}')
    
    torch.save(model.state_dict(), model_path)
    torch.save({
        'word_to_index': word_to_index,
        'index_to_word': index_to_word,
        'X_test': X_test,
        'y_test': y_test,
        'vocab_size': vocab_size
    }, 'model_metadata.pth')
    print("Model and metadata saved.")



# Prediction function
def predict_next_word(sequence, word_to_index, index_to_word, model):
    model.eval()
    sequence = [word_to_index.get(word, 0) for word in sequence.split()]
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sequence)
        return index_to_word[torch.argmax(output).item()]

# THEN HAVE THE INTERACTIVE GENERATION LOOP
# Interactive generation
print("\033[94mModel ready! Type a sequence (exit to quit):\033[0m")
while True:
    input_seq = input("\033[94mInput: \033[0m")
    if input_seq.lower() == "exit":
        break
    
    generated = input_seq.split()
    for _ in range(20):
        context = ' '.join(generated[-seq_length:])
        generated.append(predict_next_word(context, word_to_index, index_to_word, model))
    
    print(f"\033[92mGenerated: {' '.join(generated[len(input_seq.split()):])}\033[0m\n")