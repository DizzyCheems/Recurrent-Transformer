# Step 1: Install required libraries
# Run this in your terminal:
# pip install transformers torch datasets

# Step 2: Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm

# Step 3: Set device to CUDA if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 4: Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose a variant like 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 5: Set pad_token to eos_token since GPT-2 doesn't have a padding token by default
tokenizer.pad_token = tokenizer.eos_token

# Move the GPT-2 model to the GPU (or CPU if no GPU is available)
gpt2_model.to(device)

# Step 6: Define the AttentionStream model for fine-tuning
class AttentionStream(nn.Module):
    def __init__(self, vocab_size, embedding_dim=gpt2_model.config.hidden_size, hidden_dim=gpt2_model.config.hidden_size, num_layers=8, dropout=0.2, num_classes=16):
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

vocab_size = len(tokenizer)
embedding_dim = gpt2_model.config.hidden_size  # Use GPT-2's hidden size for embedding
hidden_dim = gpt2_model.config.hidden_size  # Use GPT-2's hidden size
num_layers = 8  # You can adjust this based on your requirements
dropout = 0.2
num_classes = 16

attention_stream_model = AttentionStream(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes).to(device)

# Step 7: Read and tokenize the dataset from the text file
with open("train_data.txt", "r") as file:
    text_data = file.read()

# Tokenize the text
inputs = tokenizer(text_data, return_tensors="pt", max_length=1024, truncation=True, padding=True)

# Move tokenized data to the device (GPU or CPU)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Prepare the dataset for DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Use batch_size=1 for small dataset

# Step 8: Set up the optimizer
optimizer = AdamW(attention_stream_model.parameters(), lr=5e-5)

# Step 9: Fine-tuning loop
epochs = 80  # Number of epochs to train
for epoch in range(epochs):
    attention_stream_model.train()  # Set the model to training mode
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids, attention_mask = batch

        # Forward pass
        token_output, class_output = attention_stream_model(input_ids)
        loss = nn.CrossEntropyLoss()(token_output.view(-1, vocab_size), input_ids.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

# Step 10: Save the fine-tuned model
torch.save(attention_stream_model.state_dict(), "fine_tuned_attention_stream.pth")

# Step 11: Generate text using the fine-tuned model and GPT-2 head
# Load the fine-tuned AttentionStream model
attention_stream_model.load_state_dict(torch.load("fine_tuned_attention_stream.pth"))

# Generate text
input_text = "What is the hotel check in schedule?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Perform generation using GPT-2
output = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)