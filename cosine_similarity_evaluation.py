import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os
from sklearn.model_selection import train_test_split
import ollama  # Import Ollama
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge  # For ROUGE score calculation

# Define the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
with open('test.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Tokenization and creating a word-to-index mapping
words = data.split()
word_counts = Counter(words)
word_to_index = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
index_to_word = {i: word for word, i in word_to_index.items()}

# Add an unknown token for out-of-vocabulary words
word_to_index['<UNK>'] = len(word_to_index) + 1
index_to_word[len(index_to_word) + 1] = '<UNK>'

# Encode words as indices
encoded_data = [word_to_index.get(word, word_to_index['<UNK>']) for word in words]

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
        self.mu = nn.Parameter(torch.tensor(0.5))  # Fix: Use torch.tensor instead of tectorch.tensor

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

    def get_embeddings(self, x):
        """Returns the embeddings from the model."""
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, None)
        return x[:, -1, :]  # Return the last hidden state as the embedding

# Hyperparameters
embedding_dim = 50

# Load the pre-trained model's metadata (if available)
metadata_path = 'model_metadata.pth'
if os.path.exists(metadata_path):
    metadata = torch.load(metadata_path, map_location=device, weights_only=True)  # Set weights_only=True
    vocab_size = metadata['vocab_size']
    print(f"Loaded vocabulary size from metadata: {vocab_size}")
else:
    print("No metadata found. Using vocabulary size from the dataset.")
    vocab_size = len(word_to_index) + 1
    
# Initialize the model with the correct vocabulary size
model = AttentionStreamModel(vocab_size, embedding_dim)

# Move the model to the device
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def calculate_perplexity(model, test_loader, device):
    """Calculate perplexity on test set"""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return torch.exp(torch.tensor(total_loss / len(test_loader)))

# Model loading/saving
model_path = 'attention_stream_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Set weights_only=True
    print("Loaded saved model.")
    
    # Calculate perplexity for loaded model
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"Test Perplexity: {perplexity:.2f}")
    
else:
    print("No saved model found. Please train a model first.")
    exit()
    
# Finetuning loop
epochs = 1
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
    
    print(f'Finetuning Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader)}')

# Calculate final perplexity after finetuning
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
perplexity = calculate_perplexity(model, test_loader, device)
print(f"\nFinal Test Perplexity after Finetuning: {perplexity:.2f}")

# Save finetuned model
torch.save(model.state_dict(), 'attention_stream_model_finetuned.pth')
print("Finetuned model saved.")

# Initialize ROUGE scorer
rouge = Rouge()

# Function to compute cosine similarity using AttentionStreamModel
def compute_cosine_similarity(text1, text2, model, word_to_index, seq_length=5):
    """
    Compute cosine similarity using embeddings from AttentionStreamModel.
    """
    # Tokenize and encode the input texts
    def encode_text(text):
        words = text.split()
        encoded = [word_to_index.get(word, word_to_index['<UNK>']) for word in words]
        if len(encoded) < seq_length:
            encoded += [word_to_index['<UNK>']] * (seq_length - len(encoded))  # Pad if necessary
        return torch.tensor([encoded[:seq_length]], dtype=torch.long).to(device)

    # Get embeddings for both texts
    input1 = encode_text(text1)
    input2 = encode_text(text2)
    embedding1 = model.get_embeddings(input1).detach().cpu().numpy()
    embedding2 = model.get_embeddings(input2).detach().cpu().numpy()

    # Reshape embeddings to 2D arrays
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Compute cosine similarity
    return cosine_similarity(embedding1, embedding2)[0][0]

# Function to compute ROUGE score
def compute_rouge_score(reference, hypothesis):
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]

# Testing function for cosine similarity
def test_cosine_similarity(model, word_to_index):
    """
    Test cosine similarity using AttentionStreamModel.
    """
    test_pairs = [
        ("The cat is on the mat.", "The cat is sitting on the mat."),  # Similar
        ("The cat is on the mat.", "The sun is shining brightly."),    # Dissimilar
        ("I love programming.", "Programming is my passion."),        # Similar
        ("I love programming.", "I hate bugs."),                       # Dissimilar
    ]
    
    for text1, text2 in test_pairs:
        similarity = compute_cosine_similarity(text1, text2, model, word_to_index)
        print(f"Cosine Similarity between:\n'{text1}'\nand\n'{text2}': {similarity:.4f}")

# Call the test function
test_cosine_similarity(model, word_to_index)

# Interactive generation with Ollama
print("\033[94mModel ready! Type a sequence (exit to quit):\033[0m")
while True:
    input_seq = input("\033[94mInput: \033[0m")
    if input_seq.lower() == "exit":
        break
    
    # Generate embeddings using AttentionStreamModel
    def encode_text(text):
        words = text.split()
        encoded = [word_to_index.get(word, word_to_index['<UNK>']) for word in words]
        if len(encoded) < seq_length:
            encoded += [word_to_index['<UNK>']] * (seq_length - len(encoded))  # Pad if necessary
        return torch.tensor([encoded[:seq_length]], dtype=torch.long).to(device)

    input_tensor = encode_text(input_seq)
    embedding = model.get_embeddings(input_tensor).detach().cpu().numpy()

    # Use Ollama to generate text
    generated = ollama.generate(
        model="llama3",
        prompt=input_seq,
        options={
            "max_length": 10,
            "temperature": 0.5,
        }
    )["response"]
    print(f"\033[92mGenerated: {generated}\033[0m\n")
    
    # Compute cosine similarity
    generated_embedding = model.get_embeddings(encode_text(generated)).detach().cpu().numpy()
    cosine_sim = compute_cosine_similarity(input_seq, generated, model, word_to_index)
    print(f"\033[93mCosine Similarity: {cosine_sim:.4f}\033[0m")
    
    # Compute ROUGE score
    rouge_scores = compute_rouge_score(input_seq, generated)
    print(f"\033[93mROUGE Scores: {rouge_scores}\033[0m\n")