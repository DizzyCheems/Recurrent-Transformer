# Step 1: Install required libraries
# Run this in your terminal:
# pip install transformers torch datasets

# Step 2: Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Step 3: Set device to CUDA if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 4: Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # Use a larger model if possible
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 5: Set pad_token to eos_token since GPT-2 doesn't have a padding token by default
tokenizer.pad_token = tokenizer.eos_token

# Move the GPT-2 model to the GPU (or CPU if no GPU is available)
gpt2_model.to(device)

# Step 6: Read and tokenize the updated dataset from the text file
with open("train_data.txt", "r") as file:
    text_data = file.read()

# Tokenize the text
inputs = tokenizer(text_data, return_tensors="pt", max_length=1024, truncation=True, padding=True)

# Move tokenized data to the device (GPU or CPU)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Prepare the dataset for DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Increase batch size if possible

# Step 7: Set up the optimizer
optimizer = AdamW(gpt2_model.parameters(), lr=5e-5)

# Step 8: Set up learning rate scheduler
total_steps = len(dataloader) * 16  # Update with the number of epochs you want
warmup_steps = total_steps // 10  # Warmup for the first 10% of training steps
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Step 9: Fine-tuning loop
accumulation_steps = 8  # Use gradient accumulation to simulate larger batch size
max_grad_norm = 1.0  # Gradient clipping

for epoch in range(16):  # Increase the number of epochs as necessary
    gpt2_model.train()  # Set the model to training mode
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/16")
    for batch_idx, batch in enumerate(loop):
        input_ids, attention_mask = batch

        # Forward pass
        outputs = gpt2_model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss = loss / accumulation_steps  # Scale loss for gradient accumulation

        # Backward pass
        loss.backward()

        # Gradient accumulation and optimizer step
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update the learning rate
            optimizer.zero_grad()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(gpt2_model.parameters(), max_grad_norm)

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

# Step 10: Save the fine-tuned model
gpt2_model.save_pretrained("fine_tuned_gpt2_medium")
tokenizer.save_pretrained("fine_tuned_gpt2_medium")

# Step 11: Generate text using the fine-tuned GPT-2 model
# Load the fine-tuned GPT-2 model
gpt2_model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2_medium")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_gpt2_medium")
gpt2_model.to(device)

# Generate text
input_text = "What is the hotel pet policy?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate attention mask
attention_mask = input_ids != tokenizer.pad_token_id

# Perform generation using GPT-2 with the attention mask
output = gpt2_model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
