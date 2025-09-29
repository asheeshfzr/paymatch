# app/embeddings.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict

# load model & tokenizer at startup (cached)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Precompute embeddings for all transactions
TXN_EMBEDDINGS = {}   # id -> embedding

def embed_texts(texts: List[str]) -> np.ndarray:
    """Compute embeddings for list of texts."""
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def count_tokens(text: str) -> int:
    """Return number of tokens used for input string."""
    return len(tokenizer.encode(text))

def build_txn_embeddings(transactions: Dict[str, Dict]) -> None:
    """Precompute embeddings for all transactions and store in TXN_EMBEDDINGS."""
    global TXN_EMBEDDINGS
    descriptions = []
    ids = []
    for tid, row in transactions.items():
        desc = str(row.get("description", "") or "")
        if desc:
            descriptions.append(desc)
            ids.append(tid)
    embs = embed_texts(descriptions)
    TXN_EMBEDDINGS = {tid: emb for tid, emb in zip(ids, embs)}

def search_transactions(query: str, top_k: int = 5) -> List[Dict]:
    """Return top_k most similar transactions with cosine similarity."""
    if not TXN_EMBEDDINGS:
        return []

    q_emb = embed_texts([query])[0]
    tx_ids = list(TXN_EMBEDDINGS.keys())
    tx_embs = np.array([TXN_EMBEDDINGS[tid] for tid in tx_ids])

    sims = cosine_similarity([q_emb], tx_embs)[0]
    ranked = sorted(zip(tx_ids, sims), key=lambda x: x[1], reverse=True)[:top_k]

    results = [{"id": tid, "embedding": float(score)} for tid, score in ranked]
    return results
