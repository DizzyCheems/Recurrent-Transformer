import pandas as pd

# Update the file path to point to your CSV file
csv_file = 'Conversation.csv'  # This is a CSV file, not an Excel file

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Step 2: Initialize an empty list to hold the formatted strings
formatted_data = []

# Step 3: Loop through the rows and create the required format
for index, row in df.iterrows():
    question = row['question']
    answer = row['answer']
    
    # Format the question and answer in the required format
    formatted_data.append(f"{question}\n{answer}\n\n")

# Step 4: Write the formatted data to a text file
output_file = 'formatted_output.txt'

with open(output_file, 'w') as f:
    f.writelines(formatted_data)

print(f"Formatted dataset has been saved to {output_file}")
