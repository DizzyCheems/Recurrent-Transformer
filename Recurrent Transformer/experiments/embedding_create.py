import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json

# Load the pre-trained model and tokenizer for embedding generation
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # You can change this model based on your requirements
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to convert text to embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling for sentence embeddings
    return embeddings.squeeze().numpy()  # Convert to a numpy array

# Read the responses from response.txt
def read_responses(file_path='response.txt'):
    prompts = []
    responses = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):  # Assuming each entry consists of 3 lines
            prompt = lines[i].strip().replace("Prompt: ", "")
            response = lines[i + 1].strip().replace("Response: ", "")
            prompts.append(prompt)
            responses.append(response)
    
    return prompts, responses

# Function to save embeddings as a model file (e.g., JSON or numpy file)
def save_embeddings(embeddings, file_name='embeddings_model.json'):
    # Saving as a dictionary with embeddings
    embedding_dict = {f"response_{i+1}": emb.tolist() for i, emb in enumerate(embeddings)}  # Convert numpy arrays to lists
    with open(file_name, 'w') as f:
        json.dump(embedding_dict, f, indent=4)  # Save as JSON for easy inspection and use

# Main function
def main():
    # Read responses from the file
    prompts, responses = read_responses()

    # Generate embeddings for the responses
    embeddings = []
    for response in responses:
        embedding = get_embedding(response)
        embeddings.append(embedding)

    # Save the embeddings to a file
    save_embeddings(embeddings)
    print(f"Embeddings saved successfully!")

if __name__ == '__main__':
    main()
