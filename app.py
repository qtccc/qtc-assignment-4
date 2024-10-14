from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents)

# Perform LSA (using Truncated SVD)
lsa = TruncatedSVD(n_components=100, random_state=42)
X_lsa = lsa.fit_transform(X)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Transform the query into the LSA space
    query_vec = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vec)
    
    # Calculate cosine similarity between the query and all documents
    similarities = cosine_similarity(query_lsa, X_lsa).flatten()
    
    # Get the top 5 most similar document indices
    top_indices = similarities.argsort()[-5:][::-1]
    
    # Retrieve the top 5 documents and their corresponding similarities
    top_documents = [documents[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    
    return top_documents, top_similarities, top_indices


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    # Debug: print the query received
    print(f"Received query: {query}")
    
    documents, similarities, indices = search_engine(query)
    
    # Debug: print the raw data received from search_engine
    print(f"Documents: {documents[:2]}")  # Show only first 2 documents for brevity
    print(f"Similarities: {similarities}")
    print(f"Indices: {indices}")
    
    # Ensure documents are plain strings
    documents = [str(doc) for doc in documents]

    # Convert numpy arrays to lists if necessary
    if isinstance(similarities, np.ndarray):
        similarities = similarities.tolist()
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()

    # Debug: print the final data to be returned
    print(f"Returning JSON - Documents: {documents[:2]}, Similarities: {similarities}, Indices: {indices}")
    
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})




if __name__ == '__main__':
    app.run(debug=True)
