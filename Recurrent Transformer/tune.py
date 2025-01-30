import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nltk
from collections import Counter

# Download NLTK tokenizer (if not already done)
nltk.download('punkt')

# Step 1: Tokenize Text into Words
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.tokenized_text = [nltk.word_tokenize(sentence.lower()) for sentence in text.split('\n')]  # Tokenizing the text
        self.flattened_tokens = [word for sentence in self.tokenized_text for word in sentence]
        self.vocab = sorted(set(self.flattened_tokens))
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        # Convert the entire text to word indices
        self.data = [self.word_to_idx[word] for word in self.flattened_tokens]
        
    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def load_text_data(folder_path):
    text = ''
    file_path = os.path.join(folder_path, 'qa_dataset.txt')  # Adjust to your dataset location
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Step 2: Define the AttentionStream Model
class AttentionStream(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=8, dropout=0.2, num_classes=16):
        super(AttentionStream, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Linear projections for attention mechanism
        self.attention_character = nn.Linear(hidden_dim, hidden_dim)  # Receptance vector
        self.time_modulation = nn.Parameter(torch.ones(hidden_dim))  # Weight decay vector
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)  # Key vector
        self.attention_value = nn.Linear(hidden_dim, hidden_dim)  # Value vector
        self.output_weights = nn.Linear(hidden_dim, hidden_dim)  # Output weights

        self.sequence_merging = nn.Linear(hidden_dim * 2, hidden_dim)  # For combining inputs
        self.class_merging = nn.Linear(hidden_dim * 2, hidden_dim)  # Class merging

        # Add a classification layer
        self.classification_layer = nn.Linear(hidden_dim, num_classes)

        # Layer Normalization layers
        self.layer_norm_rnn = nn.LayerNorm(hidden_dim)
        self.layer_norm_attention = nn.LayerNorm(hidden_dim)
        self.layer_norm_output = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.embedding(x)
        
        # RNN layer with residual connection and LayerNorm
        rnn_out, _ = self.rnn(x)
        rnn_out = self.layer_norm_rnn(rnn_out + x)  # Residual connection with layer norm
        rnn_out = self.dropout(rnn_out)

        # Token merging with residual connection and LayerNorm
        prev_x = torch.roll(rnn_out, shifts=1, dims=1)  # Shift tokens for token merging

        C_t = self.attention_character(rnn_out) + self.attention_character(prev_x)
        A_t = self.attention_key(rnn_out) + self.attention_key(prev_x)
        T_t = self.attention_value(rnn_out) + self.attention_value(prev_x)

        # Apply time modulation on C_t
        C_t = C_t * self.time_modulation  # Apply time modulation here

        # Sequence merging with residual connection and LayerNorm
        mixed_input = self.sequence_merging(torch.cat([C_t, A_t], dim=-1))
        mixed_input = self.layer_norm_attention(mixed_input + C_t)  # Residual connection with LayerNorm

        # Class merging with residual connection and LayerNorm
        mixed_output = self.class_merging(torch.cat([mixed_input, T_t], dim=-1))
        mixed_output = self.layer_norm_attention(mixed_output + mixed_input)  # Residual connection with LayerNorm

        # Apply time modulation again to mixed_output
        mixed_output = mixed_output * self.time_modulation  # Apply time modulation again here

        # Output gating with residual connection and LayerNorm
        ot = self.output_weights(mixed_output)
        ot = self.layer_norm_output(ot + mixed_output)  # Residual connection with LayerNorm

        # Language model output
        token_output = self.fc(ot)

        # Classification output (based on the last token's hidden state)
        class_output = self.classification_layer(rnn_out[:, -1, :])  # Use the last token's output for classification

        return token_output, class_output

# Step 3: Fine-Tuning Function
def fine_tune(model, dataset, batch_size=16, num_epochs=12, learning_rate=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            token_output, _ = model(x_batch)

            # Compute token loss
            token_loss = criterion(token_output.view(-1, dataset.vocab_size), y_batch.view(-1))
            
            loss = token_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the model after fine-tuning
    torch.save(model.state_dict(), 'fine_tuned_attention_stream.pth')

# Step 4: Text Generation Function
def generate_sequence(model, prompt, dataset, max_length=100, temperature=1.0):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prompt_tokens = nltk.word_tokenize(prompt.lower())  # Tokenize prompt and convert to word indices
    prompt_indices = [dataset.word_to_idx[word] for word in prompt_tokens if word in dataset.word_to_idx]

    # Initialize the prompt as the starting sequence
    input_sequence = torch.tensor(prompt_indices).unsqueeze(0).to(device)

    generated_sequence = prompt_tokens
    for _ in range(max_length):
        with torch.no_grad():
            token_output, _ = model(input_sequence)
            # Use temperature to control randomness (higher values = more randomness)
            token_probabilities = torch.softmax(token_output[:, -1, :] / temperature, dim=-1)
            next_token_idx = torch.multinomial(token_probabilities, 1).item()

            next_word = dataset.idx_to_word[next_token_idx]
            generated_sequence.append(next_word)

            # Update the input sequence by adding the new token
            input_sequence = torch.cat([input_sequence, torch.tensor([[next_token_idx]]).to(device)], dim=1)

    return ' '.join(generated_sequence)

# Step 5: Load and Prepare the Data
def main():
    text_data = load_text_data('text-data')  # Your data folder (replace with the actual path to the dataset)
    dataset = TextDataset(text_data, seq_length=512)

    # Initialize the model with the vocab size of the word-level dataset
    model = AttentionStream(vocab_size=dataset.vocab_size)

    # Fine-tune the model
    fine_tune(model, dataset)

    # Load the fine-tuned model
    model.load_state_dict(torch.load('fine_tuned_attention_stream.pth'))

    print("Model fine-tuned and ready to generate text!")
    print("Type 'exit' to stop the program.")
    
    while True:
        prompt = input("Enter your prompt: ")  # Take input from the user
        
        if prompt.lower() == 'exit':
            print("Exiting the program.")
            break
        
        # Generate sequence after fine-tuning
        generated_text = generate_sequence(model, prompt, dataset)
        print(f"Generated Text: {generated_text}\n")
        
        # Ask for the correct answer to simulate learning
        correct_answer = input("Enter the correct answer (or press Enter to skip): ")
        
        if correct_answer:
            # Fine-tune the model with the correct answer as additional data
            print("Fine-tuning model with corrected answer...")
            # Here, we simulate learning by manually adding this new data (in a real scenario, we could fine-tune over multiple prompts).
            augmented_text = text_data + '\n' + correct_answer  # Augment the dataset with the correct answer
            augmented_dataset = TextDataset(augmented_text, seq_length=512)  # Create a new dataset
            fine_tune(model, augmented_dataset)  # Fine-tune on the augmented dataset
            print("Model fine-tuned with the corrected answer.")

        # Continue asking for new prompts

if __name__ == '__main__':
    main()
