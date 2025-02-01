import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re

# Define the dataset
class HotelPolicyDataset(Dataset):
    def __init__(self, policies, word2idx):
        self.policies = policies
        self.word2idx = word2idx
        self.tokenized_policies = [self.tokenize(policy) for policy in policies]
    
    def tokenize(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return torch.tensor([self.word2idx[word] for word in words if word in self.word2idx])
    
    def __len__(self):
        return len(self.tokenized_policies)
    
    def __getitem__(self, idx):
        return self.tokenized_policies[idx]

# Sample dataset of hotel policies
policies = [
    "Check-in is from 3:00 PM to 11:00 PM.",
    "Check-out is from 7:00 AM to 12:00 PM.",
    "Guests can cancel free of charge until 24 hours before arrival.",
    "If the guest does not show up, they will be charged the total price of the reservation.",
    "Pets are allowed on request. Charges may apply.",
    "Smoking is not allowed in the rooms or public areas.",
    "Payment is due upon arrival. We accept credit cards and cash.",
    "All children are welcome. Children under 2 years stay free of charge in a crib.",
    "Extra beds are available upon request. Charges may apply.",
    "WiFi is available in all areas and is free of charge."
]

# Build vocabulary
all_words = re.findall(r'\b\w+\b', ' '.join(policies).lower())
vocab = Counter(all_words)
word2idx = {word: idx for idx, (word, _) in enumerate(vocab.items(), start=1)}
word2idx['<pad>'] = 0
idx2word = {idx: word for word, idx in word2idx.items()}

# Collate function to pad sequences
def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=word2idx['<pad>'])

# Define the LinearTransformer model
class LinearTransformerLanguageModel(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(LinearTransformerLanguageModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        # Convert tokens to embeddings
        embeddings = self.embedding(x)
        N, seq_len, d_model = embeddings.shape

        # Apply linear transformations
        Q = self.feature_map(self.WQ(embeddings))
        K = self.feature_map(self.WK(embeddings))
        V = self.WV(embeddings)

        # Initialize S and output
        S = torch.zeros(N, d_model, d_model, device=x.device)
        V_out = torch.zeros_like(Q)

        for i in range(seq_len):
            S = S + torch.bmm(K[:, i, :].unsqueeze(2), V[:, i, :].unsqueeze(1))
            V_out[:, i, :] = torch.bmm(Q[:, i, :].unsqueeze(1), S).squeeze(1)
        
        logits = self.fc_out(V_out)
        return logits

# Training loop
def train(model, dataloader, device, epochs=200, learning_rate=0.0001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss() 

    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.long().to(device)
            optimizer.zero_grad()
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Sequence generation function
def generate_sequence(model, prompt, device, word2idx, idx2word, max_length=50):
    model.eval()
    words = re.findall(r'\b\w+\b', prompt.lower())
    generated = torch.tensor([word2idx[word] for word in words if word in word2idx], device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_length - len(words)):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    
    generated_words = [idx2word[idx.item()] for idx in generated.squeeze()]
    return ' '.join(generated_words)

# Main script
if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = HotelPolicyDataset(policies, word2idx)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Define vocabulary size (number of unique words)
    vocab_size = len(word2idx)

    # Instantiate and train the model
    d_model = 128  # Increase the embedding size for more model capacity
    model = LinearTransformerLanguageModel(d_model, vocab_size)
    train(model, dataloader, device)

    # Generate a sequence based on a prompt
    sample_prompt = "Check-in is from"  # Starting prompt
    generated_text = generate_sequence(model, sample_prompt, device, word2idx, idx2word)
    print("Generated sequence:", generated_text)