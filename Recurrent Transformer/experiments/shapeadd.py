import torch
import torch.nn as nn
import torch.optim as optim
import ollama
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time  # For unique filenames

# ANSI escape codes for coloring
BLUE = "\033[94m"
VIOLET = "\033[95m"
RESET = "\033[0m"

# Step 1: Define multiple prompts for the TinyLlama model
prompts = [
    "How many prime numbers are within the range of 1 to 100?",
    "What is the next number in the sequence: 2, 6, 12, 20, ?",
]

# Step 2: Get multiple responses for each prompt
def get_responses_from_tinyllama(prompts, num_responses=3):
    responses = []
    
    for prompt in prompts:
        for _ in range(num_responses):  # Get multiple responses for each prompt
            response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
            generated_text = response.message.content
            responses.append((prompt, generated_text))  # Store prompt and generated response as a tuple
    
    return responses

# Step 3: Generate a large set of prompts and responses
responses = get_responses_from_tinyllama(prompts, num_responses=3)

# Step 4: Tokenize the combined responses and create a vocabulary
tokens = []
for prompt, response in responses:
    tokens.extend(prompt.split())
    tokens.extend(response.split())

vocabulary = list(set(tokens))  # Create a unique vocabulary from tokens
vocab_size = len(vocabulary)

# Map tokens to indices and vice versa
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Step 5: Convert tokens into indices
indices = []
for prompt, response in responses:
    prompt_indices = [word_to_idx[token] for token in prompt.split()]
    response_indices = [word_to_idx[token] for token in response.split()]
    indices.append((prompt_indices, response_indices))

# Step 6: Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Learnable embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)  # Get embeddings for the tokens
        out, hn = self.rnn(embedded)  # Pass through the RNN
        out = out[:, -1, :]  # Get the last hidden state from the sequence
        out = self.fc(out)  # Pass through the fully connected layer
        return out, hn  # Return the output and the hidden state for embeddings

# Step 7: Create a new model with the updated vocabulary size
embedding_dim = 50  # You can adjust this value
hidden_size = 128   # You can adjust this value
output_size = vocab_size  # Output size should match the vocabulary size
model = RNNModel(vocab_size, embedding_dim, hidden_size, output_size)

# Step 8: Function to save the model without overwriting
def save_model(model, model_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Create a timestamp for uniqueness
    unique_model_path = f"{model_path}_{timestamp}.pth"  # Append timestamp to the filename
    torch.save(model.state_dict(), unique_model_path)
    print(f"Model saved as: {unique_model_path}")

# Step 9: Generate sequences based on the RNN and temperature sampling
def generate_sequence(model, prompt, max_length=30, temperature=1.0):
    model.eval()  # Set the model to evaluation mode

    # Start with the initial prompt and convert it into indices
    tokens = prompt.split()
    indices = [word_to_idx[token] for token in tokens]
    input_tensor = torch.tensor(indices).unsqueeze(0)  # Add batch dimension (1, seq_len)

    generated_sequence = tokens  # Start with the prompt as part of the generated sequence

    for _ in range(max_length):  # Generate a sequence up to max_length
        output, hidden = model(input_tensor)  # Forward pass through the RNN

        # Apply temperature scaling to the output logits
        output = output / temperature  # Scale the logits by temperature

        # Use softmax to get probabilities for the next word
        probabilities = torch.softmax(output, dim=1)

        # Sample the next word based on the probabilities
        next_word_idx = torch.multinomial(probabilities, 1).item()

        # Get the predicted word
        predicted_word = idx_to_word[next_word_idx]

        # Append the predicted word to the sequence
        generated_sequence.append(predicted_word)

        # Update the input tensor to include the predicted word
        input_tensor = torch.tensor([next_word_idx]).unsqueeze(0)  # Update with new token

    return " ".join(generated_sequence)  # Join the sequence of words

# Step 10: Train the RNN model on the responses
def train_model(model, indices, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for prompt_indices, response_indices in indices:
            model.zero_grad()
            input_tensor = torch.tensor(prompt_indices).unsqueeze(0)  # Add batch dimension
            target_tensor = torch.tensor(response_indices).unsqueeze(0)  # Add batch dimension

            output, _ = model(input_tensor)
            loss = criterion(output, target_tensor[:, -1])  # Use the last token as target
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(indices)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Train the model on the generated responses
train_model(model, indices)

# Step 11: Save the new model with updated embeddings and fully connected layer
model_path = "new_rnn_model"
save_model(model, model_path)

# Step 12: Initialize new embeddings for the responses
def initialize_new_embeddings(model, new_vocab_size, embed_dim):
    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    new_embedding.weight.data = torch.randn(new_vocab_size, embed_dim)  # Random initialization for new vocabulary size
    return new_embedding

# Initialize new embeddings for the vocabulary size
new_vocab_size = vocab_size  # Use the size of the vocabulary created from responses
new_embedding_layer = initialize_new_embeddings(model, new_vocab_size, embedding_dim)

# Update the model's embedding layer
model.embedding = new_embedding_layer

print("Model updated with new embeddings for the new vocabulary size:", new_vocab_size)

# Example of generating sequences based on the updated model
generated_sequences = []
for prompt, _ in responses:
    generated_sequence = generate_sequence(model, prompt, max_length=30, temperature=0.7)  # Adjust temperature as needed
    generated_sequences.append(generated_sequence)

# Print out the generated sequences with colors
for i, (prompt, response) in enumerate(responses):
    print(f"{BLUE}Prompt {i+1}: {prompt}{RESET}")
    print(f"{BLUE}TinyLlama Response {i+1}: {response}{RESET}")

for i, seq in enumerate(generated_sequences):
    print(f"{VIOLET}Generated Sequence {i+1}: {seq}{RESET}")