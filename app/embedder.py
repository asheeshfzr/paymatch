# app/embedder.py
from typing import List, Optional
import numpy as np

# try sentence-transformers and tokenizer
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    S_T_AVAILABLE = True
except Exception:
    S_T_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.use_st = False
        self.tokenizer = None
        self.fitted = False
        self.corpus_embs = None
        
        if S_T_AVAILABLE:
            try:
                print(f"INFO: Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    print(f"INFO: Loaded tokenizer for {model_name}")
                except Exception as e:
                    print(f"WARNING: Failed to load tokenizer for {model_name}: {e}")
                    self.tokenizer = None
                self.use_st = True
                print(f"INFO: Successfully initialized sentence transformer")
            except Exception as e:
                print(f"ERROR: Failed to load sentence transformer {model_name}: {e}")
                self.use_st = False
        else:
            print("WARNING: sentence-transformers not available, using TF-IDF fallback")

        # Initialize TF-IDF fallback
        try:
            self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
            self.svd = TruncatedSVD(n_components=128, random_state=42)
        except Exception as e:
            print(f"ERROR: Failed to initialize TF-IDF components: {e}")
            self.tfidf = None
            self.svd = None
        self.corpus_ids = []

    def fit_corpus(self, ids: List[str], texts: List[str]):
        """Fit the embedder on a corpus of texts with error handling."""
        try:
            if not ids or not texts:
                print("WARNING: Empty corpus provided to fit_corpus")
                return
            
            if len(ids) != len(texts):
                print(f"WARNING: Mismatch between ids ({len(ids)}) and texts ({len(texts)}) length")
                return
            
            self.corpus_ids = ids
            
            if self.use_st and self.model:
                try:
                    embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                    self.corpus_embs = embs
                    self.fitted = True
                    print(f"INFO: Fitted sentence transformer on {len(texts)} texts")
                    return
                except Exception as e:
                    print(f"ERROR: Sentence transformer encoding failed: {e}")
                    self.use_st = False
            
            # Fallback to TF-IDF
            if self.tfidf and self.svd:
                try:
                    X = self.tfidf.fit_transform(texts)
                    Xd = self.svd.fit_transform(X)
                    norms = np.linalg.norm(Xd, axis=1, keepdims=True)
                    norms[norms==0] = 1.0
                    Xd = Xd / norms
                    self.corpus_embs = Xd
                    self.fitted = True
                    print(f"INFO: Fitted TF-IDF on {len(texts)} texts")
                except Exception as e:
                    print(f"ERROR: TF-IDF fitting failed: {e}")
                    self.fitted = False
            else:
                print("ERROR: No embedding method available")
                self.fitted = False
                
        except Exception as e:
            print(f"ERROR: fit_corpus failed: {e}")
            self.fitted = False

    def embed_texts(self, texts: List[str]):
        """Embed a list of texts with error handling."""
        try:
            if not texts:
                print("WARNING: Empty text list provided to embed_texts")
                return np.array([])
            
            if self.use_st and self.model:
                try:
                    return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                except Exception as e:
                    print(f"ERROR: Sentence transformer encoding failed: {e}")
                    self.use_st = False
            
            # Fallback to TF-IDF
            if self.tfidf and self.svd:
                try:
                    X = self.tfidf.transform(texts)
                    Xd = self.svd.transform(X)
                    norms = np.linalg.norm(Xd, axis=1, keepdims=True)
                    norms[norms==0] = 1.0
                    Xd = Xd / norms
                    return Xd
                except Exception as e:
                    print(f"ERROR: TF-IDF transform failed: {e}")
                    return np.array([])
            else:
                print("ERROR: No embedding method available")
                return np.array([])
                
        except Exception as e:
            print(f"ERROR: embed_texts failed: {e}")
            return np.array([])

    def embed_query(self, text: str):
        """Embed a single query text with error handling."""
        try:
            if not text or not text.strip():
                print("WARNING: Empty query text provided")
                return np.zeros(128)  # Return zero vector as fallback
            
            embeddings = self.embed_texts([text])
            if len(embeddings) > 0:
                return embeddings[0]
            else:
                print("WARNING: No embeddings returned for query")
                return np.zeros(128)  # Return zero vector as fallback
                
        except Exception as e:
            print(f"ERROR: embed_query failed: {e}")
            return np.zeros(128)  # Return zero vector as fallback

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with error handling."""
        try:
            if not text:
                return 0
            
            if self.tokenizer:
                try:
                    toks = self.tokenizer.encode(text, add_special_tokens=False)
                    return len(toks)
                except Exception as e:
                    print(f"WARNING: Tokenizer failed, using word count: {e}")
                    return max(1, len(str(text).split()))
            else:
                return max(1, len(str(text).split()))
                
        except Exception as e:
            print(f"ERROR: count_tokens failed: {e}")
            return max(1, len(str(text).split())) if text else 0
