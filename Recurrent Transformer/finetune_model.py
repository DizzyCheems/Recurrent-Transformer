import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from recurrentGPT import AttentionStream, TextDataset, load_text_data  # Importing the model and dataset from pretrain_model.py

# Adjust the vocabulary size of your new dataset to match the pre-trained model's vocab size (optional)
def adjust_vocab_size(dataset, desired_vocab_size=75):  # Update to 75 for alignment with pre-trained model
    if dataset.vocab_size > desired_vocab_size:
        print(f"Truncating vocab size from {dataset.vocab_size} to {desired_vocab_size}...")
        # Truncate the vocabulary to match the desired size (use the most frequent tokens)
        dataset.vocab = sorted(dataset.vocab)[:desired_vocab_size]
    elif dataset.vocab_size < desired_vocab_size:
        print(f"Padding vocab size from {dataset.vocab_size} to {desired_vocab_size}...")
        # Add padding tokens (e.g., "<PAD>" token) to match the desired size
        additional_tokens = ["<PAD>"] * (desired_vocab_size - dataset.vocab_size)
        dataset.vocab += additional_tokens

    # Update the vocab and corresponding indices
    dataset.char_to_idx = {char: idx for idx, char in enumerate(dataset.vocab)}
    dataset.idx_to_char = {idx: char for char, idx in dataset.char_to_idx.items()}
    dataset.vocab_size = len(dataset.vocab)

    return dataset

def fine_tune_model(model, dataset, seq_length=512, batch_size=4, num_epochs=3, learning_rate=0.0001, model_file='attention_stream_model_finetuned.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            token_output, class_output = model(x_batch)

            token_loss = criterion(token_output.view(-1, dataset.vocab_size), y_batch.view(-1))
            class_labels = torch.zeros(x_batch.size(0), dtype=torch.long).to(device)
            class_loss = criterion(class_output, class_labels)

            loss = token_loss + class_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                token_output, class_output = model(x_batch)
                token_loss = criterion(token_output.view(-1, dataset.vocab_size), y_batch.view(-1))
                val_loss += token_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

    # Save the fine-tuned model
    torch.save(model.state_dict(), model_file)

def main():
    seq_length = 512
    batch_size = 4
    num_epochs = 5
    learning_rate = 0.0001
    model_file = 'attention_stream_model_finetuned.pth'  # Save the fine-tuned model with a new name

    # Load your dataset
    text_data = load_text_data('text-data')
    dataset = TextDataset(text_data, seq_length)

    # Adjust the vocabulary size if needed (optional)
    desired_vocab_size = 75  # Set the desired vocab size (same as pre-trained model's vocab size)
    dataset = adjust_vocab_size(dataset, desired_vocab_size)

    # Load the pretrained model
    model = AttentionStream(vocab_size=dataset.vocab_size)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if os.path.exists('attention_stream_model.pth'):
        print("Pretrained model found. Loading model...")
        model.load_state_dict(torch.load('attention_stream_model.pth'), strict=False)  # Skip mismatched layers
    else:
        print("Pretrained model not found, exiting.")
        return

    fine_tune_model(model, dataset, seq_length, batch_size, num_epochs, learning_rate, model_file)

if __name__ == '__main__':
    main()
