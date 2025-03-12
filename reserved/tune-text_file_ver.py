import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import math
import torch.nn.functional as F

# Step 1: Read and tokenize data from text files in "text-data" folder
data_folder = "text-data"
vocab = []

for filename in glob.glob(os.path.join(data_folder, "*.txt")):
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().split()
        vocab.extend(text)

# Remove duplicate tokens and sort the vocabulary
vocab = sorted(set(vocab))

# Create a word-to-index dictionary
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

# Step 2: Define the GPT model

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_sequence_length, num_heads, dropout_rate, weight_decay):
        super(GPT, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._create_positional_encoding(max_sequence_length)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Added normalization layer

        self.dropout = nn.Dropout(dropout_rate)
        self.weight_decay = weight_decay

    def _create_positional_encoding(self, max_sequence_length):
        pe = torch.zeros(1, max_sequence_length, self.embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embedding_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        embedded = self.embedding(x)
        batch_size, sequence_length, _ = embedded.size()
        embedded = embedded + self.positional_encoding[:, :sequence_length, :]

        # Permute dimensions for multihead attention
        embedded = embedded.permute(1, 0, 2)

        # Apply multihead attention
        attn_output, _ = self.multihead_attention(embedded, embedded, embedded)

        # Reshape and permute dimensions for GRU
        attn_output = attn_output.permute(1, 0, 2).contiguous()
        attn_output = attn_output.view(batch_size, sequence_length, -1)

        # Apply GRU with dropout and normalization
        attn_output = self.dropout(attn_output)
        output, _ = self.gru(attn_output)
        output = self.layer_norm(output)  # Apply normalization

        # Apply linear layer with weight decay
        output = F.linear(output, self.linear.weight, self.linear.bias)
        l2_regularization = self.weight_decay * torch.norm(self.linear.weight)
        output += l2_regularization

        return output

# Set hyperparameters
embedding_dim = 160
hidden_dim = 800
lr = 0.0001
epochs = 180
batch_size = 128
max_sequence_length = 80
num_heads = 8

dropout_rate = 0.1  # Set your desired dropout rate value
weight_decay = 0.01  # Set your desired weight decay value
model = GPT(vocab_size, embedding_dim, hidden_dim, max_sequence_length, num_heads, dropout_rate, weight_decay)

# Step 3: Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train_model():
    data_folder = ""
    filename = "pairs.txt"

    with open(os.path.join(data_folder, filename), "r", encoding="utf-8", errors="replace") as f:
        text = f.readlines()

    # Split each line into input and target sequences
    pairs = [line.strip().split("/") for line in text]
    inputs, targets = zip(*pairs)

    # Tokenize inputs and targets
    input_tokens = [sentence.split() for sentence in inputs]
    target_tokens = [sentence.split() for sentence in targets]

    # Extend the vocabulary with input and target tokens
    vocab.extend([word for sentence in input_tokens for word in sentence])
    vocab.extend([word for sentence in target_tokens for word in sentence])

    # Create a word-to-index dictionary
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    vocab_size = len(vocab)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate random indices for training
        indices = torch.randint(0, len(input_tokens), (batch_size,))

        # Create input and target sequences based on the random indices
        sequences = [input_tokens[i] for i in indices]
        targets = [target_tokens[i] for i in indices]

        # Convert sequences to tensor format
        sequences = [[word2idx[word] for word in sequence] for sequence in sequences]
        targets = [[word2idx[word] for word in target] for target in targets]

        # Pad sequences to max_sequence_length
        sequences = [sequence + [0] * (max_sequence_length - len(sequence)) for sequence in sequences]
        targets = [target + [0] * (max_sequence_length - len(target)) for target in targets]

        sequences = torch.tensor(sequences)
        targets = torch.tensor(targets)

        # Forward pass
        output = model(sequences)

        # Calculate loss and perform backpropagation
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        optimizer.step()

        # Print input and output tensors
        print(f"Epoch {epoch+1}:")
        for i in range(batch_size):
            print("Input sequence:", [idx.item() for idx in sequences[i]])
            print("Output sequence:", [idx.item() for idx in torch.argmax(output[i], dim=-1)])
            print("Loss:", loss.item())
            print()

        # Save the model checkpoint
        torch.save(model.state_dict(), "gpt_model.pt")

def nucleus_sampling(probabilities, p):
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
    probabilities[mask] = 0
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    return probabilities

def generate_autocompletion(prompt, max_length=20, temperature=1.0, p=0.9):
    prompt_sequence = torch.tensor([[word2idx[word] for word in prompt.split()]])

    with torch.no_grad():
        generated_sentence = prompt.split()

        for _ in range(max_length):
            # Forward pass
            output = model(prompt_sequence)

            # Apply temperature and nucleus sampling to the output probabilities
            scaled_output = output[:, -1, :] / temperature
            probabilities = torch.softmax(scaled_output, dim=-1)
            probabilities = nucleus_sampling(probabilities, p)

            # Sample a word from the probability distribution
            last_word_idx = torch.multinomial(probabilities, num_samples=1).item()
            last_word = idx2word[last_word_idx]
            generated_sentence.append(last_word)

            if last_word == '<EOS>':
                break

            # Update the prompt sequence
            prompt_sequence = torch.cat((prompt_sequence, torch.tensor([[last_word_idx]])), dim=1)

    return generated_sentence


# Check if a pre-trained model exists
pretrained_model_exists = True  # Set this to True if you have a pre-trained model
model_checkpoint_path = "gpt_model.pt"  # Provide the path to your pre-trained model checkpoint

# Ask user for option to continue training, generate a sentence, or use the pre-trained model
while True:
    choice = input("Enter '1' to continue training, '2' to generate autocompletions (type 'exit' to quit): ")

    if choice == "1":
        if pretrained_model_exists and os.path.exists(model_checkpoint_path):
            # Load the pre-trained model state dictionary
            pretrained_dict = torch.load(model_checkpoint_path)

            # Remove the missing keys from the state dictionary
            pretrained_dict.pop("layer_norm.weight", None)
            pretrained_dict.pop("layer_norm.bias", None)

            # Load the modified state dictionary into the model
            model.load_state_dict(pretrained_dict, strict=False)

        train_model()
    elif choice == "2":
        if pretrained_model_exists and os.path.exists(model_checkpoint_path):
            # Load the pre-trained model state dictionary
            pretrained_dict = torch.load(model_checkpoint_path)

            # Remove the missing keys from the state dictionary
            pretrained_dict.pop("layer_norm.weight", None)
            pretrained_dict.pop("layer_norm.bias", None)

            # Load the modified state dictionary into the model
            model.load_state_dict(pretrained_dict, strict=False)
            while True:
                prompt = input("Enter a prompt (type 'exit' to go back to the main menu): ")
                if prompt == "exit":
                    break
                temperature = 2.0  # Adjust this value to control the randomness
                p = 0.9  # Adjust this value for nucleus sampling
                generated_sentence = generate_autocompletion(prompt, temperature=temperature, p=p)
                print("Generated Autocompletion:", " ".join(generated_sentence))
        else:
            print("Pre-trained model not found. Please choose option 1 to continue training.")
        break
    else:
        print("Invalid choice. Please enter '1', '2' or 'exit'.")
