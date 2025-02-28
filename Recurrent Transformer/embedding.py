import torch
import torch.nn as nn
import ollama

# Step 1: Get the text response from tinyllama for the prompt
prompt = "What animals are llamas related to?"

# Using ollama.chat to get the generated response text
response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])

# Print the entire response object to inspect its structure (for debugging purposes)
print("Full Response from tinyllama:", response)

# Extract the generated text from the response
generated_text = response.message.content
print("Generated Response Text from tinyllama:", generated_text)

# Now let's also get the embeddings for the generated response
response_embedding = ollama.embed(model="tinyllama", input=generated_text)
response_embedding = torch.tensor(response_embedding['embeddings'][0]).unsqueeze(0)  # Add batch dimension

# Step 2: Define a simple RNN in PyTorch
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # The input tensor must have the shape (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)  # Pass through the RNN
        out = out[:, -1, :]  # Get the last hidden state from the sequence
        out = self.fc(out)  # Pass through the fully connected layer
        return out

# Step 3: Instantiate the model
input_size = response_embedding.shape[-1]  # The size after embedding (should match the embedding size)
hidden_size = 128  # You can adjust this
output_size = 10  # Example: Number of output classes (you can change this)
model = SimpleRNN(input_size, hidden_size, output_size)

# Step 4: Define a simple vocabulary (for demonstration)
vocabulary = ["cat", "dog", "llama", "sheep", "horse", "cow", "bird", "fish", "tiger", "elephant"]

# Step 5: Reshape the embedding to have a sequence length of 1
response_embedding = response_embedding.unsqueeze(1)  # Now the shape will be (1, 1, input_size)

# Step 6: Pass the reshaped response embedding through the RNN
output = model(response_embedding.float())

# Step 7: Convert the RNN output to a word by finding the index of the highest value
_, predicted_idx = torch.max(output, dim=1)  # Get the index with the highest value
predicted_word = vocabulary[predicted_idx.item()]  # Map the index to a word

# Step 8: Print the predicted word
print(predicted_word)
