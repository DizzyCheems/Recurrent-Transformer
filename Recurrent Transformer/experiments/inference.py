import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import math  # Import the math module

# Load the embeddings from the JSON file
def load_embeddings(file_path='embeddings_model.json'):
    with open(file_path, 'r') as f:
        embedding_dict = json.load(f)
    # Convert to numpy arrays (each response embedding)
    embeddings = [np.array(embedding) for embedding in embedding_dict.values()]
    return np.array(embeddings)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Perform inference
def perform_inference(model, embeddings, device):
    # Convert embeddings to tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(embeddings_tensor)
    
    return outputs

# Main function to load embeddings, initialize model, and perform inference
def main():
    # Load embeddings
    embeddings = load_embeddings('embeddings_model.json')
    
    # Model input/output dimensions
    input_dim = embeddings.shape[1]  # Number of features per embedding
    hidden_dim = 128  # Size of hidden layer
    output_dim = 2  # For binary classification (you can adjust this for your use case)

    # Initialize the model
    model = SimpleNN(input_dim, hidden_dim, output_dim)

    # Assuming you have a pre-trained model, load the model weights here if needed
    # model.load_state_dict(torch.load("model_weights.pth"))
    
    # For simplicity, we use random weights in this example
    model.apply(init_weights)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Perform inference on the loaded embeddings
    outputs = perform_inference(model, embeddings, device)
    
    # Print the raw outputs (logits or class probabilities)
    print("Inference Outputs:", outputs)

    # For classification, you can apply a softmax and get class probabilities
    # Assuming it's a binary classification task
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    print("Class Probabilities:", probabilities)

    # Example: Print the predicted class (0 or 1 for binary classification)
    predicted_classes = torch.argmax(probabilities, dim=1)
    print("Predicted Classes:", predicted_classes)

# Function to initialize weights (optional for better model performance)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if __name__ == "__main__":
    main()