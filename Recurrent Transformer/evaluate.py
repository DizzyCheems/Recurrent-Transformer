import torch
from AttentionStream import AttentionStreamModel

# Load metadata
metadata = torch.load('model_metadata.pth', map_location='cpu')
word_to_index = metadata['word_to_index']
index_to_word = metadata['index_to_word']
vocab_size = metadata['vocab_size']
seq_length = 5  # Should match training sequence length

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionStreamModel(vocab_size, embedding_dim=50).to(device)
model.load_state_dict(torch.load('attention_stream_model.pth', map_location=device))
model.eval()

def predict_next_word(context, temperature=1.0):
    """Generate next word given a context"""
    # Convert context to tensor
    words = context.split()
    if len(words) != seq_length:
        raise ValueError(f"Context must be exactly {seq_length} words")
    
    encoded = [word_to_index.get(word, 0) for word in words]
    input_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output / temperature, dim=1)
        next_idx = torch.argmax(probs, dim=1).item()
    
    return index_to_word.get(next_idx, '<UNK>')

def generate_text(seed, max_length=20, temperature=1.0):
    """Generate sequence starting with seed"""
    generated = seed.split()
    for _ in range(max_length):
        context = ' '.join(generated[-seq_length:])
        if len(context.split()) < seq_length:
            # Pad with special token if needed
            context = '<PAD> ' * (seq_length - len(context.split())) + context
        next_word = predict_next_word(context, temperature)
        generated.append(next_word)
    return ' '.join(generated)

# Interactive generation loop
print("\n\033[94mModel loaded! Enter a seed sequence of exactly 5 words (type 'exit' to quit):\033[0m")
while True:
    try:
        seed = input("\033[94mSeed sequence: \033[0m").strip().lower()
        if seed == 'exit':
            break
        
        if len(seed.split()) != seq_length:
            print(f"\033[91mError: Input must be exactly {seq_length} words.\033[0m")
            continue
        
        generated_text = generate_text(seed)
        print(f"\033[92mGenerated text: {generated_text}\033[0m\n")
    
    except ValueError as e:
        print(f"\033[91mError: {e}\033[0m")
    except Exception as e:
        print(f"\033[91mAn error occurred: {str(e)}\033[0m")