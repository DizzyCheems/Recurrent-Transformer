import ollama  # Import Ollama

# Function to interact with TinyLlama model
def get_tinyllama_response(prompt: str):
    # Chat with TinyLlama model
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    
    # Print the entire response to check its structure (for debugging)
    print("Raw Response:", response)
    
    # Try to extract the correct key, assuming the key is 'message' (check and update based on your observation)
    if 'message' in response and 'content' in response['message']:
        return response['message']['content'].replace('\n', ' ').strip()  # Extract and clean the content
    elif 'text' in response:
        return response['text'].replace('\n', ' ').strip()
    else:
        return "No response text found"

# Sample prompts
prompts = [
    "What is the fundamental principle of counting?",
    "What is the multiplication rule in probability?",
    "Can you explain the addition rule in probability?",
    "What is the difference between permutations and combinations?",
    "How do you calculate the probability of an event?",
    "What is a sample space in probability?",
    "Can you describe what a factorial is and how it's used?",
    "What is the concept of expected value in probability?",
    "How do you determine the number of ways to arrange a set of objects?",
    "What is the significance of the binomial theorem?",
    "Can you explain the concept of conditional probability?",
    "What is the law of large numbers?",
    "How do you calculate combinations and permutations?",
    "What is the role of counting in statistics?",
    "Can you provide an example of a real-world application of counting principles?",
    "What is the concept of independence in probability?",
    "How does the central limit theorem relate to counting?",
    "What are some common counting techniques used in combinatorics?",
    "Can you explain the pigeonhole principle?",
    "What is the difference between discrete and continuous probability distributions?",
    "How do you use counting to solve problems in game theory?"
]

# Collect responses from TinyLlama model for each prompt
responses = [get_tinyllama_response(prompt) for prompt in prompts]

# Save responses to the text file in the desired format using UTF-8 encoding
with open("tinyllama_responses.txt", "w", encoding="utf-8") as file:
    for response in responses:
        file.write(f"{response}\n")  # Write each response on a single line

print("Responses have been saved to 'tinyllama_responses.txt'")

# Optionally, print out the processed responses for validation
for response in responses:
    print(f"Processed Response: {response}")