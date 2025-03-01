import torch
import torch.nn as nn
import torch.optim as optim
import ollama
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time  # For unique filenames

# Step 1: Define a single prompt for the TinyLlama model
prompt = "Write me a paragrah about wolves"

# Step 2: Get a single response from TinyLlama
def get_response_from_tinyllama(prompt):
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    generated_text = response.message.content
    return generated_text

# Step 3: Get the response from TinyLlama
tinyllama_response = get_response_from_tinyllama(prompt)

# Step 4: Tokenize the prompt and response to create a vocabulary
tokens = prompt.split() + tinyllama_response.split()
vocabulary = list(set(tokens))  # Create a unique vocabulary from tokens
vocab_size = len(vocabulary)

# Map tokens to indices and vice versa
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Step 5: Convert tokens into indices
prompt_indices = [word_to_idx[token] for token in prompt.split()]
response_indices = [word_to_idx[token] for token in tinyllama_response.split()]

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

# Step 10: Train the RNN model on the response
def train_model(model, prompt_indices, response_indices, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.zero_grad()
        input_tensor = torch.tensor(prompt_indices).unsqueeze(0)  # Add batch dimension
        target_tensor = torch.tensor(response_indices).unsqueeze(0)  # Add batch dimension

        output, _ = model(input_tensor)
        loss = criterion(output, target_tensor[:, -1])  # Use the last token as target
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train the model on the generated response
train_model(model, prompt_indices, response_indices)

# Step 11: Save the new model with updated embeddings and fully connected layer
model_path = "new_rnn_model"
save_model(model, model_path)

# Step 12: Generate a sequence using the RNN model
rnn_sequence = generate_sequence(model, prompt, max_length=30, temperature=0.7)

# Step 13: Print the generated sequences with color coding
# TinyLlama's output in blue
print("\033[94mGenerated by TinyLlama:\033[0m", tinyllama_response)
# RNN's output in violet
print("\033[95mGenerated by RNN:\033[0m", rnn_sequence)