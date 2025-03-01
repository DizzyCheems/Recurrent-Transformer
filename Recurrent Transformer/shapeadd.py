import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Step 1: Prepare the Data
class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=10):
        self.seq_length = seq_length
        with open(file_path, 'r') as f:
            text = f.read()
        
        # Create vocabulary
        self.vocab = self.build_vocab(text)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        # Convert text to sequences of indices
        self.data = self.text_to_sequences(text)
    
    def build_vocab(self, text):
        words = text.split()
        word_counts = Counter(words)
        vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        vocab.insert(0, '<UNK>')  # Add unknown token at the beginning
        return vocab
    
    def text_to_sequences(self, text):
        words = text.split()
        sequences = []
        for i in range(0, len(words) - self.seq_length):
            seq = words[i:i + self.seq_length]
            sequences.append([self.word_to_idx.get(word, 0) for word in seq])  # Use <UNK> for unknown words
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

# Step 2: Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        out = self.fc(output[:, -1, :])
        return out

# Step 3: Train the Model
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch[:, -1])
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 4: Save the Model
def save_model(model, vocab, file_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, file_path)

# Function to predict the next word
def predict_next_word(model, input_seq, word_to_idx):
    model.eval()
    input_indices = []
    
    for word in input_seq:
        word = word.lower()  # Ensure case consistency
        input_indices.append(word_to_idx.get(word, 0))  # Use <UNK> for unknown words

    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
    return dataset.idx_to_word[predicted_idx]

# Load the model for inference
def load_model(file_path):
    checkpoint = torch.load(file_path, weights_only=True)
    model = RNNModel(len(checkpoint['vocab']), embed_dim, hidden_dim, len(checkpoint['vocab']))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['vocab']

# Initialize new embeddings
def initialize_new_embeddings(model, new_vocab_size, embed_dim):
    # Create a new embedding layer with random weights
    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    
    # Initialize the new weights randomly
    new_embedding.weight.data = torch.randn(new_vocab_size , embed_dim)  # Random initialization for new vocabulary size
    
    return new_embedding

# Main
if __name__ == "__main__":
    # Hyperparameters
    embed_dim = 64
    hidden_dim = 128
    n_layers = 1
    seq_length = 10
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Load dataset
    dataset = TextDataset('train.txt', seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    vocab_size = len(dataset.vocab)
    model = RNNModel(vocab_size, embed_dim, hidden_dim, vocab_size, n_layers)  # Output dim is vocab size

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs)

    # Save the model
    save_model(model, dataset.vocab, 'rnn_model.pth')

    # Example of loading the model
    model, vocab = load_model('rnn_model.pth')
    print("Model loaded successfully with vocabulary size:", len(vocab))

    # Now you can use the model for predictions with the updated vocabulary
    input_sequence = ['this', 'is', 'an', 'example']  # Ensure these words are in the new vocabulary
    next_word = predict_next_word(model, input_sequence, {word: idx for idx, word in enumerate(vocab)})
    print(f'The predicted next word is: {next_word}')

    # Initialize new embeddings for a new vocabulary size
    new_vocab_size = 150  # Example new vocabulary size
    new_embedding_layer = initialize_new_embeddings(model, new_vocab_size, embed_dim)
    
    # Update the model's embedding layer
    model.embedding = new_embedding_layer
    
    print("Model updated with new embeddings for the new vocabulary size:", new_vocab_size)