# Step 1: Install required libraries
# Run this in your terminal:
# pip install transformers torch datasets

# Step 2: Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load GPT-2 model and tokenizer
model_name = "gpt2-medium"
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
gpt2_model.to(device)

# Read and tokenize data
with open("train_data.txt", "r") as file:
    text_data = file.read()

inputs = tokenizer(text_data, return_tensors="pt", max_length=512, truncation=True, padding=True)  # Reduced max_length
inputs = {key: value.to(device) for key, value in inputs.items()}

# Prepare dataset and dataloader
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Reduced batch size

# Optimizer
optimizer = AdamW(gpt2_model.parameters(), lr=5e-5)

# Mixed precision setup
scaler = GradScaler()

# Fine-tuning loop with gradient accumulation
accumulation_steps = 4

for epoch in range(25):  # Increased epochs
    gpt2_model.train()
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/25")

    for step, batch in enumerate(loop):
        input_ids, attention_mask = batch
        optimizer.zero_grad()

        with autocast():
            outputs = gpt2_model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        scaler.scale(loss).backward()
    
        # Update weights every `accumulation_steps`
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loop):
            scaler.step(optimizer)
            scaler.update()

        loop.set_postfix(loss=loss.item())

# Save the fine-tuned model
gpt2_model.save_pretrained("fine_tuned_gpt2_medium")
tokenizer.save_pretrained("fine_tuned_gpt2_medium")

# Load the fine-tuned GPT-2 model
gpt2_model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2_medium")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_gpt2_medium")
gpt2_model.to(device)

# Infinite loop for user input prompts
while True:
    input_text = input("Enter a prompt (or type 'exit' to stop): ")
    if input_text.lower() == 'exit':
        print("Exiting...")
        break

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Generate attention mask
    attention_mask = input_ids != tokenizer.pad_token_id

    # Perform generation using GPT-2 with the attention mask
    output = gpt2_model.generate(input_ids, attention_mask=attention_mask, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}\n")
