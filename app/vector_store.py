"""
Vector store and search utilities for the Invoice Reimbursement System.
Handles embedding, FAISS index management, and combined metadata/similarity search.
"""
import faiss  # Facebook AI Similarity Search for fast vector search
import numpy as np
from sentence_transformers import SentenceTransformer  # For text embeddings
import os
import pickle
import re

# Paths for storing FAISS index and metadata
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
FAISS_META_PATH = os.path.join(os.path.dirname(__file__), "faiss_metadata.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model

# Load or initialize FAISS index and metadata
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        metadata_list = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(384)  # 384 is the embedding size for all-MiniLM-L6-v2
    metadata_list = []

def add_invoice_analysis_to_vector_db(analyses):
    """
    Adds a list of invoice analysis dicts to the FAISS vector store.
    Each dict should have: invoice_id, status, reason, employee_name, date, full_text
    Embeds the analysis and updates the index and metadata.
    """
    global faiss_index, metadata_list
    for analysis in analyses:
        # Combine invoice text and analysis for embedding
        text_to_embed = analysis["full_text"] + "\nStatus: " + analysis["status"] + "\nReason: " + analysis["reason"]
        embedding = embedding_model.encode(text_to_embed)
        embedding = np.array([embedding]).astype('float32')
        faiss_index.add(embedding)
        metadata_list.append({
            "invoice_id": analysis["invoice_id"],
            "status": analysis["status"],
            "reason": analysis["reason"],
            "employee_name": analysis["employee_name"],
            "date": analysis["date"]
        })
    # Persist index and metadata to disk
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(metadata_list, f)

def search_invoices(query, top_k=5, filters=None):
    """
    Search the FAISS index for invoices similar to the query string, optionally filtering by metadata.
    Returns a list of (score, metadata) tuples.
    """
    if len(metadata_list) == 0:
        return []
    # Apply metadata filters if provided
    filtered_indices = list(range(len(metadata_list)))
    if filters:
        def match(meta):
            # Check if all filter values are present in metadata (case-insensitive)
            for k, v in filters.items():
                if k in meta and v.lower() not in str(meta[k]).lower():
                    return False
            return True
        filtered_indices = [i for i, meta in enumerate(metadata_list) if match(meta)]
    if not filtered_indices:
        return []
    # Prepare filtered embeddings and metadata
    filtered_embeddings = faiss_index.reconstruct_n(0, len(metadata_list))
    filtered_embeddings = [filtered_embeddings[i] for i in filtered_indices]
    filtered_embeddings = np.array(filtered_embeddings).astype('float32')
    # Encode query
    query_vec = embedding_model.encode(query)
    query_vec = np.array([query_vec]).astype('float32')
    # Build a temporary FAISS index for filtered subset
    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)
    D, I = temp_index.search(query_vec, min(top_k, len(filtered_indices)))
    results = []
    for idx, score in zip(I[0], D[0]):
        real_idx = filtered_indices[idx]
        results.append((score, metadata_list[real_idx]))
    return results

def extract_metadata_filters(query):
    """
    Extracts possible metadata filters from the user query using simple keyword matching.
    Returns a dict with keys: status, employee_name, date (if found).
    """
    filters = {}
    # Status keywords for matching
    status_keywords = {
        "declined": "Declined",
        "reimbursed": "Fully Reimbursed",
        "partial": "Partially Reimbursed",
        "partially reimbursed": "Partially Reimbursed",
        "fully reimbursed": "Fully Reimbursed"
    }
    for k, v in status_keywords.items():
        if k in query.lower():
            filters["status"] = v
            break
    # Simple employee name extraction (look for 'for <name>' or 'by <name>')
    name_match = re.search(r"(?:for|by) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", query)
    if name_match:
        filters["employee_name"] = name_match.group(1)
    # Simple date extraction (YYYY-MM-DD or YYYY/MM/DD)
    date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", query)
    if date_match:
        filters["date"] = date_match.group(1)
    return filters 