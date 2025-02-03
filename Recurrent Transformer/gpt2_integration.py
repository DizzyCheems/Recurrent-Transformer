import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the end token
END_TOKEN = "<|endoftext|>"  # Define the special end token
# Define color codes
BLUE = '\033[94m'  # Blue color for user input
GREEN = '\033[92m'  # Green color for generated text
RESET = '\033[0m'  # Reset to default color

# Adjustments: Loading GPT-2 model and tokenizer
class HybridModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=8, dropout=0.2, num_classes=16):
        super(HybridModel, self).__init__()
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


def load_text_data(folder_path):
    text = ''
    file_path = os.path.join(folder_path, 'qa_dataset.txt')  # Adjusted for your dataset
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        
        # Append the end token to the end of the text (optional, for fine-tuning purposes)
        self.text_with_end_token = text + " " + END_TOKEN
        self.tokenized_data = tokenizer.encode(self.text_with_end_token)

    def __len__(self):
        return len(self.tokenized_data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.tokenized_data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.tokenized_data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y


# Initialize GPT-2 model and tokenizer
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add the special end token to the tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': [END_TOKEN]})
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    # Resize token embeddings to include the new end token
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def generate_gpt2_prompt(model, tokenizer, prompt_text, max_length=100):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    input_ids = input_ids[:, :1024]  # Ensure the input sequence doesn't exceed 1024 tokens
    
    # Set do_sample=True to allow temperature sampling
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, 
                                do_sample=True, temperature=0.7, no_repeat_ngram_size=2, 
                                eos_token_id=tokenizer.encode(END_TOKEN)[0])
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text



def generate_text(model, start_text, tokenizer, device, length=150):
    model.eval()
    input_text = tokenizer.encode(start_text)
    input_seq = torch.tensor(input_text, dtype=torch.long).unsqueeze(0).to(device)  # Move tensor to device
    generated = start_text
    
    for _ in range(length):
        with torch.no_grad():
            token_output, _ = model(input_seq)
        probabilities = torch.softmax(token_output[:, -1], dim=-1)
        next_token_idx = torch.multinomial(probabilities, num_samples=1).item()
        
        next_token = tokenizer.decode([next_token_idx])
        
        # If the next token is the end token, stop the generation
        if next_token == END_TOKEN:
            break
        
        generated += next_token
        input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[next_token_idx]], device=device)), dim=1)
    
    return generated




def main():
    seq_length = 512  # Adjusted for question-answer pairs
    batch_size = 4    # Adjusted batch size for smaller dataset
    num_epochs = 5   # Increased for better training
    learning_rate = 0.0001
    model_file = 'hybrid_attention_stream_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GPT-2 tokenizer and model
    gpt2_model, gpt2_tokenizer = load_gpt2_model()

    text_data = load_text_data('text-data')
    dataset = TextDataset(text_data, gpt2_tokenizer, seq_length)

    model = HybridModel(vocab_size=dataset.vocab_size)
    model = model.to(device)

    # Check if the model already exists
    if os.path.exists(model_file):
        print("Model found. Loading model...")
        model.load_state_dict(torch.load(model_file, weights_only=True))
    else:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                token_output, class_output = model(x_batch)

                # Compute loss for both token generation and classification
                token_loss = criterion(token_output.view(-1, dataset.vocab_size), y_batch.view(-1))
                class_labels = torch.zeros(x_batch.size(0), dtype=torch.long).to(device)
                class_loss = criterion(class_output, class_labels)

                loss = token_loss + class_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Save the model
        torch.save(model.state_dict(), model_file)

    # User input for generating text
    while True:
        user_input = input(f"{BLUE}Enter the starting text (or 'exit' to quit): {RESET}")
        if user_input.lower() == 'exit':
            break
        if len(user_input) > seq_length:
            user_input = user_input[:seq_length]

        print("\n\n")

        # Generate text using GPT-2 as the assistant to the AttentionStream model
        gpt2_generated = generate_gpt2_prompt(gpt2_model, gpt2_tokenizer, user_input, max_length=50)
        
        # Use AttentionStream to complete the generated sequence
        generated_text = generate_text(model, gpt2_generated, gpt2_tokenizer, device, length=150)
        
        print(f"{GREEN}{generated_text}{RESET}")


if __name__ == '__main__':
    main()
