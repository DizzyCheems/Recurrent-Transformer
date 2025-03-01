import torch
import os

# Function to load model state dicts
def load_model_state_dicts(model_paths):
    state_dicts = []
    for path in model_paths:
        state_dict = torch.load(path)
        state_dicts.append(state_dict)
    return state_dicts

# Function to merge embeddings
def merge_embeddings(state_dicts):
    # Determine the maximum vocabulary size
    max_vocab_size = max(state_dict['embedding.weight'].size(0) for state_dict in state_dicts)
    embed_dim = state_dicts[0]['embedding.weight'].size(1)  # Assuming all models have the same embedding dimension

    # Initialize a new embedding tensor with the maximum vocabulary size
    merged_embeddings = torch.zeros(max_vocab_size, embed_dim)

    # Average the embeddings from all models
    for state_dict in state_dicts:
        current_vocab_size = state_dict['embedding.weight'].size(0)
        # Copy existing weights into the merged embeddings
        merged_embeddings[:current_vocab_size, :] += state_dict['embedding.weight']

    # Average the embeddings
    merged_embeddings /= len(state_dicts)

    return merged_embeddings

# Function to save the new model with merged embeddings
def save_merged_model(merged_embeddings, model_path, vocab_size, embed_dim):
    # Create a new model class (you can define it here or import it)
    class RNNModel(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(RNNModel, self).__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.rnn = torch.nn.RNN(embedding_dim, 128, batch_first=True)  # Hidden size can be adjusted
            self.fc = torch.nn.Linear(128, vocab_size)

        def forward(self, x):
            embedded = self.embedding(x)
            out, _ = self.rnn(embedded)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    # Create a new model instance
    new_model = RNNModel(vocab_size, merged_embeddings.size(1))
    new_model.embedding.weight.data = merged_embeddings

    # Save the new model
    torch.save(new_model.state_dict(), model_path)
    print(f"Merged model saved as: {model_path}")

if __name__ == "__main__":
    # List of model paths to merge
    model_paths = [
        "new_rnn_model_20250301-150251.pth",
        "new_rnn_model_20250301-150317.pth",
        "new_rnn_model_20250301-150412.pth"
    ]

    # Load the state dicts
    state_dicts = load_model_state_dicts(model_paths)

    # Merge the embeddings
    merged_embeddings = merge_embeddings(state_dicts)

    # Save the new model with merged embeddings
    vocab_size = merged_embeddings.size(0)  # Use the maximum vocab size
    embed_dim = merged_embeddings.size(1)  # Use the embedding dimension
    save_merged_model(merged_embeddings, "merged_rnn_model.pth", vocab_size, embed_dim)