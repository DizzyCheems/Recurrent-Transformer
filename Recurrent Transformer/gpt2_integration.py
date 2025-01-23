from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn

class AttentionStream(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=8, dropout=0.2, num_classes=16):
        super(AttentionStream, self).__init__()
    
        # Load GPT-2 tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set padding token to the eos_token (common practice for GPT-2)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

        # GPT-2 embedding dimension (768 for GPT-2 small)
        self.gpt2_embedding_dim = self.gpt2_model.config.n_embd

        # If you want to project GPT-2's embedding size (768) to a smaller dimension
        self.embedding_projection = nn.Linear(self.gpt2_embedding_dim, embedding_dim)

        # RNN layer
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
        # Tokenize the input text (x is a batch of sentences or texts)
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)

        # Get GPT-2 embeddings for the input text (output shape: [batch_size, seq_len, 768])
        gpt2_embeddings = self.gpt2_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]

        # Optionally project the GPT-2 embeddings to the desired embedding dimension (embedding_dim)
        embedded_input = self.embedding_projection(gpt2_embeddings)  # Shape: [batch_size, seq_len, embedding_dim]

        # Pass the embeddings through the RNN
        rnn_out, _ = self.rnn(embedded_input)
        rnn_out = self.layer_norm_rnn(rnn_out + embedded_input)  # Residual connection with LayerNorm
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

        # Language model output (predictions for tokens)
        token_output = self.fc(ot)

        # Classification output (based on the last token's hidden state)
        class_output = self.classification_layer(rnn_out[:, -1, :])  # Use the last token's output for classification

        return token_output, class_output, inputs

# Sampling function for text generation
def generate_text(model, prompt, max_length=50, temperature=1.0, top_k=50):
    # Tokenize the input prompt
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")

    # Use GPT-2 for text generation with top-k and temperature sampling
    output = model.gpt2_model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,    # Control the randomness (higher means more random)
        top_k=top_k,                # Top-k sampling (consider top-k most probable tokens)
        top_p=0.95,                 # Nucleus sampling (top-p sampling)
        do_sample=True,             # Enable sampling
        num_return_sequences=1      # Number of output sequences to generate
    )

    # Decode the generated tokens to text
    generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage:
model = AttentionStream(vocab_size=5000)  # Assuming vocab_size is 5000 (adjust as needed)

# Generate text using the sampling function
input_prompt = "What are the Hotel Policies?"
generated_text = generate_text(model, input_prompt, max_length=50, temperature=0.8, top_k=50)

print("Generated Text:")
print(generated_text)
