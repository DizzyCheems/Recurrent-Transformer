import torch
from torch.utils.data import TensorDataset, DataLoader
from model.attention_stream import AttentionStreamModel
from metrics import MetricCalculator

# Load metadata
metadata = torch.load('model_metadata.pth')
word_to_index = metadata['word_to_index']
index_to_word = metadata['index_to_word']
X_test = metadata['X_test']
y_test = metadata['y_test']
vocab_size = metadata['vocab_size']

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load model
    model = AttentionStreamModel(vocab_size, embedding_dim=50)
    model.load_state_dict(torch.load("attention_stream_model.pth", map_location=device))
    model.to(device)
    
    # Create test dataset
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize calculator
    calculator = MetricCalculator(
        model=model,
        word_to_index=word_to_index,
        index_to_word=index_to_word,
        X_test=X_test,
        y_test=y_test,
        device=device
    )
    
    # Calculate metrics
    print("Calculating Perplexity...")
    ppl = calculator.calculate_perplexity(test_loader)
    print(f"Perplexity: {ppl:.2f}")
    
    print("\nCalculating Cosine Similarity...")
    cos_sim = calculator.calculate_cosine_sim(test_loader)
    print(f"Cosine Similarity: {cos_sim:.4f}")
    
    print("\nCalculating ROUGE-L...")
    rouge = calculator.calculate_rouge(test_samples=100)
    print(f"ROUGE-L F1: {rouge:.4f}")