import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Load the data
with open('data.txt', 'r') as file:
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

# Define the RNN-based model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(RNNModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embeddings(x)  # Shape: [batch_size, seq_length, embedding_dim]
        rnn_out, _ = self.rnn(embedded)  # RNN outputs (output, hidden)
        out = self.fc(rnn_out[:, -1, :])  # Take the output of the last RNN timestep
        return out

# Hyperparameters
embedding_dim = 50
hidden_dim = 128
vocab_size = len(word_to_index) + 1  # Add 1 for padding
model = RNNModel(vocab_size, embedding_dim, hidden_dim)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)  # Forward pass
        loss = criterion(output, batch_y)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}')

# Function to predict the next word
def predict_next_word(sequence, word_to_index, index_to_word, model):
    model.eval()
    sequence = [word_to_index.get(word, 0) for word in sequence.split()]
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence)
        predicted_index = torch.argmax(output, dim=1).item()  # Get the word index with highest probability
    return index_to_word[predicted_index]

# Interactive loop for generating sequences
print("Model is ready! Type a sequence of words, and the model will predict the next word.")
print("Type 'exit' to stop.")

while True:
    input_sequence = input("Enter a sequence: ")
    if input_sequence.lower() == "exit":
        break
    
    num_predictions = int(input("How many words would you like the model to predict? "))
    
    generated_sequence = input_sequence.split()
    
    for _ in range(num_predictions):
        next_word = predict_next_word(' '.join(generated_sequence[-seq_length:]), word_to_index, index_to_word, model)
        generated_sequence.append(next_word)
        print(f"Predicted next word: {next_word}")
    
    print(f"Generated sequence: {' '.join(generated_sequence)}\n")
