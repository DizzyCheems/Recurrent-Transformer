# Path to the input and output files
input_file = 'human_chat.txt'  # Replace with your input text file path
output_file = 'formatted_data_output.txt'  # Output file path

# Read the conversation from the input text file with UTF-8 encoding
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Initialize a list to hold the formatted data
formatted_data = []

# Iterate over the lines and extract questions and answers
for i in range(0, len(lines) - 1, 2):  # Jump by 2 to get pairs of lines
    # Clean up the question line (remove "Human 2:" prefix)
    question = lines[i].strip().replace("Human 2:", "").strip()

    # Clean up the answer line (remove "Human 1:" prefix)
    answer = lines[i + 1].strip().replace("Human 1:", "").strip()

    # Format the question and answer pair
    formatted_data.append(f"{question}\n{answer}\n\n")

# Write the formatted data to a text file
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(formatted_data)

print(f"Formatted dataset has been saved to {output_file}")
