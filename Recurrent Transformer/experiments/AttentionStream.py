import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from collections import Counter

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to read responses from the text file
def read_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        responses = file.readlines()
    return [response.strip() for response in responses if response.strip()]

# Read responses from the text file
responses = read_responses('tinyllama_responses.txt')

# Tokenization and vocabulary creation
def tokenize(texts):
    words = []
    for text in texts:
        words.extend(text.lower().split())
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(words).items())}
    vocab['<PAD>'] = 0  # Adding padding token
    return vocab

vocab = tokenize(responses)

# Convert text to sequences
def text_to_sequence(text, vocab):
    return [vocab.get(word.lower(), 0) for word in text.split()]

# Prepare the data
sequences = [text_to_sequence(response, vocab) for response in responses]
max_len = max(len(seq) for seq in sequences)

# Padding sequences
padded_sequences = [seq + [vocab['<PAD>']] * (max_len - len(seq)) for seq in sequences]

# Create input and target tensors
X_train = torch.tensor([seq[:-1] for seq in padded_sequences], dtype=torch.long).to(device)  # Input (all but last token)
y_train = torch.tensor([seq[1:] for seq in padded_sequences], dtype=torch.long).to(device)  # Target (all but first token)

# Create a Dataset and DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

# Define the new model classes
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
        merged = self .sequence_merging(shifted, shifted, shifted)
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
        return self.fc(x)  # Return the output for all time steps

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 80
num_layers = 1  # You can increase this for deeper models

# Instantiate the model and move it to the device
model = AttentionStreamModel(vocab_size, embedding_dim, num_layers).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))  # Flatten for loss calculation
        loss.backward()
        
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Generate text based on the learned responses
def generate_text(model, vocab, start_word, max_len=100, temperature=1.0):
    model.eval()
    input_sequence = text_to_sequence(start_word, vocab)
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # Move to device
    
    generated_words = []
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_tensor)
            output = output[:, -1, :] / temperature  # Apply temperature
            probabilities = F.softmax(output, dim=1)  # Convert to probabilities
            predicted_index = torch.multinomial(probabilities, num_samples=1).item()  # Sample from the distribution
            
            predicted_word = list(vocab.keys())[list(vocab.values()).index(predicted_index)]
            generated_words.append(predicted_word)
            
            # Update input tensor with the predicted word
            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_index]], device=device)), dim=1)  # Shift left and append predicted word
    
    return ' '.join(generated_words)

# Generate text based on a starting word with temperature
start_word = "calculate the probability of an event"
generated_text = generate_text(model, vocab, start_word, temperature=0.7)  # Adjust temperature
print(f"Generated text: {generated_text}")