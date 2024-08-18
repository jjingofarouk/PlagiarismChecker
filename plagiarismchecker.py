import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertModel
import torch

# Load SpaCy model for NLP tasks
nlp = spacy.load('en_core_web_sm')

# Load BERT tokenizer and model for paraphrase detection
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to clean and preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    return text

# Function to calculate cosine similarity between two documents
def calculate_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]

# Function to calculate n-gram similarity
def ngram_similarity(doc1, doc2, n=3):
    vectorizer = CountVectorizer(ngram_range=(n, n)).fit([doc1, doc2])
    ngram_matrix = vectorizer.transform([doc1, doc2])
    similarity_matrix = cosine_similarity(ngram_matrix[0:1], ngram_matrix[1:2])
    return similarity_matrix[0][0]

# Function to calculate Levenshtein Distance similarity
def levenshtein_similarity(doc1, doc2):
    return SequenceMatcher(None, doc1, doc2).ratio()

# Function to get BERT embeddings for sentences
def get_bert_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Function to calculate BERT-based semantic similarity
def semantic_similarity(doc1, doc2):
    emb1 = get_bert_embedding(doc1)
    emb2 = get_bert_embedding(doc2)
    return cosine_similarity(emb1, emb2)[0][0]

# Main function to check plagiarism
def sophisticated_plagiarism_detector(document, corpus):
    document = preprocess(document)
    results = {}

    for idx, doc in enumerate(corpus):
        processed_doc = preprocess(doc)

        cosine_sim = calculate_similarity(document, processed_doc)
        ngram_sim = ngram_similarity(document, processed_doc)
        levenshtein_sim = levenshtein_similarity(document, processed_doc)
        semantic_sim = semantic_similarity(document, processed_doc)

        # Weighted average of different similarities
        overall_similarity = (0.25 * cosine_sim + 0.25 * ngram_sim +
                              0.25 * levenshtein_sim + 0.25 * semantic_sim)

        results[f'Document {idx+1}'] = overall_similarity

    return results

# Example usage
if __name__ == "__main__":
    # Document to check for plagiarism
    doc_to_check = input("Enter the text you want to check for plagiarism")


    # Corpus of documents to compare against
    corpus = [
        "First document to compare against.",
        "Second document to compare against.",
        "Third document to compare against."
    ]

    # Detect plagiarism
    results = sophisticated_plagiarism_detector(doc_to_check, corpus)

    # Display the results
    for doc, similarity in results.items():
        print(f"{doc}: {similarity * 100:.2f}% similarity")
