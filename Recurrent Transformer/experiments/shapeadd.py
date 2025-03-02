import ollama  # Import Ollama

# Sample prompts (new set of prompts)
prompts = [
    "What is Bayes' Theorem?",
    "Explain the law of total probability.",
]

# Open the 'response.txt' file in append mode (to add to the file, not overwrite it)
with open('response.txt', 'a') as file:
    # Loop through each prompt and generate a response using TinyLlama
    for prompt in prompts:
        # Generate response from the TinyLlama model
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
        
        # Print the response to inspect its structure (for debugging)
        print(response)
        
        # Check if the response contains a 'text' key or use the correct key
        if 'text' in response:
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Response: {response['text']}\n\n")
        else:
            # Handle case when 'text' is not found, print the whole response
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Response (no 'text' key): {response}\n\n")
