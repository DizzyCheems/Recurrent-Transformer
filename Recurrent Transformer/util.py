import spacy
from spacy.matcher import Matcher
import pandas as pd

# Ensure the spaCy model is downloaded
# !python -m spacy download en_core_web_trf

# Load the spaCy transformer model
nlp = spacy.load('en_core_web_trf')

def process_text(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Tokenization
    print("Tokens:")
    for token in doc:
        print(token.text, end=' | ')

    # Part-of-Speech Tagging and Lemmatization
    print("\n\nPOS Tags and Lemmas:")
    for token in doc:
        print(f"{token.text} ({token.pos_}, {token.lemma_})")

    # Named Entity Recognition
    print("\nNamed Entities:")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")

    # Dependency Parsing
    print("\nDependencies:")
    dependency_data = []
    for token in doc:
        print(f"{token.text} --> {token.dep_} --> {token.head.text}")
        dependency_data.append({
            "Token": token.text,
            "Dependency": token.dep_,
            "Head": token.head.text
        })

    # Create a DataFrame from the dependency data
    df = pd.DataFrame(dependency_data)
    print("\nDependency Summary Table:")
    print(df)

    # Identify subjects and their corresponding verbs, and generate responses
    responses = []
    template = "The subject [SUBJECT] performs the action [VERB]."
    
    print("\nSubjects and their Verbs:")
    for token in doc:
        if token.dep_ == "nsubj":
            # Find the verb associated with the subject
            verb = token.head
            response = template.replace("[SUBJECT]", token.text).replace("[VERB]", verb.text)
            responses.append(response)
            print(f"Subject: {token.text} (Verb: {verb.text})")

    # Print the generated responses
    print("\nGenerated Responses:")
    for response in responses:
        print(response)

    # Similarity
    doc2 = nlp("Apple is a technology company.")
    similarity = doc.similarity(doc2)
    print(f"\nSimilarity between the documents: {similarity:.4f}")

    # Find matches using the Matcher
    matcher = Matcher(nlp.vocab)
    pattern = [{"LEMMA": "buy"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN"}]
    matcher.add("BUY_PATTERN", [pattern])

    matches = matcher(doc)
    print("\nMatches found:")
    for match_id, start, end in matches:
        span = doc[start:end]
        print(span.text)

# Main program to input a sentence and process it
def main():
    while True:
        user_input = input("\nEnter a sentence to process (or type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break
        process_text(user_input)

if __name__ == "__main__":
    main()