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

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Check if a saved model exists
model_path = 'rnn_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
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