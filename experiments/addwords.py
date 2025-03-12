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
    "Count from 1 to 20.",
    "List the numbers from 5 to 15.",
    "Can you count backward from 10 to 1?",
    "Enumerate the first ten even numbers.",
    "Count the odd numbers between 1 and 30.",
    "What are the numbers from 1 to 50 in increments of 5?",
    "Can you list the prime numbers up to 30?",
    "Count the numbers in the range of 1 to 100 that are divisible by 3.",
    "How many numbers are there between 1 and 100?",
    "Can you count the multiples of 7 from 1 to 70?",
    "List the numbers from 1 to 10 in reverse order.",
    "Count the total number of digits in the numbers from 1 to 100.",
    "What are the first ten Fibonacci numbers?",
    "Can you count the numbers that are both even and greater than 10?",
    "How many integers are there between -10 and 10?",
    "Count the numbers from 1 to 100 that end with the digit 5.",
    "Can you list the numbers from 1 to 12 and their squares?",
    "Count the total number of vowels in the numbers from 1 to 20 when spelled out.",
    "What is the sum of all numbers from 1 to 50?",
    "Can you explain how to count the total number of outcomes when rolling two dice?",
    "How do you count the number of ways to select 3 items from a set of 10?"
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