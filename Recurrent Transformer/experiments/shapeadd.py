import torch
import torch.nn as nn
import torch.optim as optim
import ollama
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time  # For unique filenames

# Step 1: Define a list of logical reasoning prompts
prompts = [
"If two cars start from the same point and travel in opposite directions, how far apart will they be after 1 hour if one car travels at 50 mph and the other at 60 mph?"
"How many months have 28 days?"
"A man is pushing his car along a road when he comes to a hotel. He shouts, 'I'm bankrupt!' Why?"
"What comes once in a minute, twice in a moment, but never in a thousand years?"
]

# Step 2: Get a response from TinyLlama for each prompt
def get_response_from_tinyllama(prompt):
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    generated_text = response.message.content
    return generated_text

# Step 3: Get the responses from TinyLlama for each prompt
responses = [get_response_from_tinyllama(prompt) for prompt in prompts]

# Step 4: Tokenize the prompts and responses to create a vocabulary
tokens = []
for prompt, response in zip(prompts, responses):
    tokens.extend(prompt.split() + response.split())
vocabulary = list(set(tokens))  # Create a unique vocabulary from tokens
vocab_size = len(vocabulary)

# Map tokens to indices and vice versa
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Step 5: Convert prompts and responses into indices
prompt_indices = [ [word_to_idx[token] for token in prompt.split()] for prompt in prompts ]
response_indices = [ [word_to_idx[token] for token in response.split()] for response in responses ]

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
        out = self.fc(out)  # Pass through the fully connected layer
        return out, hn  # Return the output and the hidden state

# Step 7: Create a new model with the updated vocabulary size
embedding_dim = 2048  # You can adjust this value
hidden_size = 512   # You can adjust this value
output_size = vocab_size  # Output size should match the vocabulary size
model = RNNModel(vocab_size, embedding_dim, hidden_size, output_size)

# Step 8: Function to save the model without overwriting
def save_model(model, model_path):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Create a timestamp for uniqueness
    unique_model_path = f"{model_path}_{timestamp}.pth"  # Append timestamp to the filename
    torch.save(model.state_dict(), unique_model_path)
    print(f"Model saved as: {unique_model_path}")

# Step 9: Generate sequences based on the RNN and temperature sampling
def generate_sequence(model, prompt, max_length=100, temperature=1.0):
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
        probabilities = torch.softmax(output[:, -1, :], dim=1)

        # Sample the next word based on the probabilities
        next_word_idx = torch.multinomial(probabilities, 1).item()

        # Get the predicted word
        predicted_word = idx_to_word[next_word_idx]

        # Append the predicted word to the sequence
        generated_sequence.append(predicted_word)

        # Update the input tensor to include the predicted word
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_word_idx]])], dim=1)

    return " ".join(generated_sequence)  # Join the sequence of words

# Step 10: Prepare training data as (input, target) pairs
def prepare_data(data_indices, seq_length=5):
    input_data = []
    target_data = []
    for i in range(len(data_indices) - seq_length):
        input_data.append(data_indices[i:i + seq_length])  # Previous words (input)
        target_data.append(data_indices[i + 1:i + seq_length + 1])  # Next word (target)
    return input_data, target_data

# Step 11: Train the RNN model on the response
# Step 11: Train the RNN model on the response
def train_model(model, prompt_indices, response_indices, num_epochs=20, learning_rate=0.001, seq_length=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Combine prompt and response indices
    all_indices = prompt_indices + response_indices

    # Prepare input-output pairs
    input_data, target_data = prepare_data(all_indices, seq_length)
    
    # Convert input and target data into tensors
    input_tensor = torch.tensor(input_data)  # Shape: (batch_size, seq_length)
    target_tensor = torch.tensor(target_data)  # Shape: (batch_size, seq_length)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_seq, target in zip(input_tensor, target_tensor):
            optimizer.zero_grad()
            input_seq = input_seq.unsqueeze(0)  # Add batch dimension (1, seq_length)

            # Forward pass
            output, _ = model(input_seq)

            # The output shape is (batch_size, seq_length, vocab_size)
            # We need to calculate the loss for each word in the sequence.
            # So we need to reshape the output and target tensors appropriately.

            # Reshape the output and target tensors
            output = output.view(-1, vocab_size)  # Flatten output to (batch_size * seq_length, vocab_size)
            target = target.view(-1)  # Flatten target to (batch_size * seq_length,)

            # Calculate loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(input_tensor)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Step 12: Train the model on the generated responses
for prompt_idx, prompt in enumerate(prompts):
    print(f"Training on prompt {prompt_idx+1}: {prompt}")
    train_model(model, prompt_indices[prompt_idx], response_indices[prompt_idx])

# Step 13: Save the new model with updated embeddings and fully connected layer
model_path = "new_rnn_model"
save_model(model, model_path)

# Step 14: Generate sequences using the RNN model
for prompt_idx, prompt in enumerate(prompts):
    print(f"\nGenerating sequence for prompt {prompt_idx+1}: {prompt}")
    rnn_sequence = generate_sequence(model, prompt, max_length=100, temperature=0.7)
    print(f"Generated by RNN: {rnn_sequence}")
