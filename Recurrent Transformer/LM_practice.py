import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os

# Load the data
with open('data.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Tokenization and creating a word-to-index mapping
words = data.split()
word_counts = Counter(words)
word_to_index = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}  # Start indexing from 1
index_to_word = {i: word for word, i in word_to_index.items()}

# Encode words as indices
encoded_data = [word_to_index[word] for word in words]

# Prepare input-output pairs (context, next word)
seq_length = 5  # Define the length of the context
X = []
y = []
for i in range(len(encoded_data) - seq_length):
    X.append(encoded_data[i:i+seq_length])  # Context (previous words)
    y.append(encoded_data[i+seq_length])   # Next word (target)

X = np.array(X)
y = np.array(y)

# Convert X and y to torch tensors
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Prepare DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class TimeMix(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeMix, self).__init__()
        self.embedding_dim = embedding_dim
        self.decay_net = nn.Linear(embedding_dim, 1)
    
    def forward(self, K, V, W):
        T = K.size(1)
        exp_k = torch.exp(K - (T - torch.arange(T, device=K.device).unsqueeze(0).unsqueeze(2)) * W)
        numerator = torch.sum(exp_k * V, dim=1)
        denominator = torch.sum(exp_k, dim=1)
        return numerator / denominator

class TimeMixSeq(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeMixSeq, self).__init__()
        self.embedding_dim = embedding_dim
        self.decay_net = nn.Linear(embedding_dim, 1)
        self.register_buffer('a', torch.zeros(1))
        self.register_buffer('b', torch.zeros(1))
    
    def forward(self, k, v, w):
        a = torch.exp(w) * self.a + torch.exp(k)
        b = torch.exp(w) * self.b + torch.exp(k) * v
        self.a, self.b = a, b
        return b / a

class ChannelMix(nn.Module):
    def __init__(self, input_dim):
        super(ChannelMix, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu2 = lambda x: torch.relu(x) ** 2

    def forward(self, x):
        return self.sigmoid(self.Wr(x)) * (self.Wv(self.relu2(self.Wk(x))))

class Shift(nn.Module):
    def __init__(self, input_dim):
        super(Shift, self).__init__()
        self.mu = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, x_prev):
        if x_prev is None:
            return x
        return self.mu * x + (1 - self.mu) * x_prev

class RWKVBlock(nn.Module):
    def __init__(self, input_dim):
        super(RWKVBlock, self).__init__()
        self.shift = Shift(input_dim)
        self.time_mix = TimeMix(input_dim)
        self.channel_mix = ChannelMix(input_dim)

    def forward(self, x, x_prev):
        shifted_input = self.shift(x, x_prev)
        time_mix_out = self.time_mix(shifted_input, shifted_input, shifted_input)
        channel_mix_out = self.channel_mix(time_mix_out)

        # Ensure channel_mix_out has the same sequence length as shifted_input
        # Reshape channel_mix_out to match shifted_input's shape
        if channel_mix_out.size(1) != shifted_input.size(1):
            # Add an extra dimension for the sequence length (broadcast over seq_length)
            channel_mix_out = channel_mix_out.unsqueeze(1).expand(-1, shifted_input.size(1), -1)

        # Now that both tensors have the same shape, we can safely add them
        return shifted_input + channel_mix_out


class RWKVModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(RWKVModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rwkv_blocks = nn.ModuleList([RWKVBlock(embedding_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embeddings(x)
        x_prev = None
        for rwkv_block in self.rwkv_blocks:
            embedded = rwkv_block(embedded, x_prev)
            x_prev = embedded
        out = self.fc(embedded[:, -1, :])  # Output of the last timestep
        return out

# Hyperparameters
embedding_dim = 50
hidden_dim = 128
vocab_size = len(word_to_index) + 1  # Add 1 for padding
model = RWKVModel(vocab_size, embedding_dim, hidden_dim)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Check if a saved model exists
model_path = 'rwkv_model.pth'
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded saved model.")
else:
    # Training the model
    epochs = 55
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move data to GPU if available
            optimizer.zero_grad()
            output = model(batch_X)  # Forward pass
            loss = criterion(output, batch_y)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

# Function to predict the next word
def predict_next_word(sequence, word_to_index, index_to_word, model):
    model.eval()
    sequence = [word_to_index.get(word, 0) for word in sequence.split()]
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension and move to GPU if available
    with torch.no_grad():
        output = model(sequence)
        predicted_index = torch.argmax(output, dim=1).item()  # Get the word index with highest probability
    return index_to_word[predicted_index]

# Number of words to predict
num_predictions = 20

# Interactive loop for generating sequences
print("\033[94mModel is ready! Type a sequence of words, and the model will predict the next word.\033[0m")
print("\033[94mType 'exit' to stop.\033[0m")

while True:
    input_sequence = input("\033[94mEnter a sequence: \033[0m")
    if input_sequence.lower() == "exit":
        break
    
    generated_sequence = input_sequence.split()
    
    for _ in range(num_predictions):
        next_word = predict_next_word(' '.join(generated_sequence[-seq_length:]), word_to_index, index_to_word, model)
        generated_sequence.append(next_word)
    
    generated_text = ' '.join(generated_sequence[len(input_sequence.split()):])
    print(f"  \033[92mGenerated sequence: {generated_text}\033[0m\n")