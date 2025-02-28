import torch
import torch.nn as nn
import ollama
import random
import numpy as np

# Step 1: Get the text response from tinyllama for the prompt
prompt = "What animals are llamas related to?"

# Using ollama.chat to get the generated response text
response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])

# Print the entire response object to inspect its structure (for debugging purposes)
print("Full Response from tinyllama:", response)

# Extract the generated text from the response
generated_text = response.message.content
print("Generated Response Text from tinyllama:", generated_text)

# Concatenate the prompt and generated response
combined_text = prompt + " " + generated_text
print("Combined Text:", combined_text)

# Step 2: Tokenize the combined text to create a vocabulary
tokens = combined_text.split()  # Split into tokens (words)
vocabulary = list(set(tokens))  # Create a unique vocabulary from tokens
vocab_size = len(vocabulary)

# Map tokens to indices and vice versa
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Step 3: Convert tokens into indices
indices = [word_to_idx[token] for token in tokens]

# Convert the indices into a tensor to feed into the RNN
input_tensor = torch.tensor(indices).unsqueeze(0)  # Add batch dimension (1, seq_len)

# Step 4: Define a simple RNN-based model
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

# Step 5: Instantiate the model
embedding_dim = 50  # You can adjust this value
hidden_size = 128   # You can adjust this value
output_size = vocab_size  # Output size should match the vocabulary size
model = RNNModel(vocab_size, embedding_dim, hidden_size, output_size)

# Step 6: Pass the input tensor through the model
output, hidden = model(input_tensor)

# Function to generate a sequence based on the prompt with temperature sampling
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

# Step 7: Generate a sequence from the RNN model with temperature sampling
generated_text_from_rnn = generate_sequence(model, combined_text, max_length=30, temperature=0.7)  # Generate up to 30 words
print("Generated Text from RNN:", generated_text_from_rnn)
