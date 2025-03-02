import ollama  # Import Ollama

# Function to interact with TinyLlama model
def get_tinyllama_response(prompt: str):
    # Chat with TinyLlama model
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    
    # Print the entire response to check its structure (for debugging)
    print("Raw Response:", response)
    
    # Try to extract the correct key, assuming the key is 'message' (check and update based on your observation)
    if 'message' in response:
        return response['message']
    elif 'text' in response:
        return response['text']
    else:
        return "No response text found"

# Function to colorize the response blue in a terminal (if running in a console)
def color_response_blue(response: str):
    return f"\033[94mTinyLlama:\n\n{response}\033[0m"

# Sample prompt
prompts = [
    "What is the fundamental principle of counting?",
    "What is the multiplication rule in probability?",
]

# Collect responses from TinyLlama model for each prompt
responses = [get_tinyllama_response(prompt) for prompt in prompts]

# Print out and colorize the responses
for response in responses:
    print(color_response_blue(response))
