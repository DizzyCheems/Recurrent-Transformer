# Step 1: Install required libraries
# Run this in your terminal:
# pip install transformers torch datasets

# Step 2: Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm

# Step 3: Set device to CUDA if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 4: Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can choose a variant like 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 5: Set pad_token to eos_token since GPT-2 doesn't have a padding token by default
tokenizer.pad_token = tokenizer.eos_token

# Move the model to the GPU (or CPU if no GPU is available)
model.to(device)

# Step 6: Read and tokenize the dataset from the text file
with open("train_data.txt", "r") as file:
    text_data = file.read()

# Tokenize the text
inputs = tokenizer(text_data, return_tensors="pt", max_length=1024, truncation=True, padding=True)

# Move tokenized data to the device (GPU or CPU)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Prepare the dataset for DataLoader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Use batch_size=1 for small dataset

# Step 7: Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 8: Fine-tuning loop
epochs = 25  # Number of epochs to train
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids, attention_mask = batch

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

# Step 9: Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

# Step 10: Generate text using the fine-tuned model
# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_gpt2")

# Move the model to the GPU (or CPU if no GPU is available)
model.to(device)

# Generate text
input_text = "What is the Hotel Check In Policy?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate predictions
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)