import os
import re

def create_word_index(file_path):
  """
  Creates an index for words in a given text file, excluding words with punctuation.

  Args:
    file_path: Path to the text file.

  Returns:
    A dictionary where keys are unique words and values are their index.
  """

  word_index = {}
  index = 0
  with open(file_path, 'r') as file:
    for line in file:
      words = re.findall(r'\b\w+\b', line.lower())  # Extract words without punctuation
      for word in words:
        if word not in word_index:
          word_index[word] = index
          index += 1
  return word_index

# Path to the text file
file_path = os.path.join('text-data', 'qa_dataset.txt') 

# Create the word index
word_index = create_word_index(file_path)

# Print the word index (for verification)
print(word_index)