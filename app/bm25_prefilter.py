# app/bm25_prefilter.py
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class BM25Prefilter:
    def __init__(self, max_features: int = 5000):
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
        self.ids = []
        self.mat = None

    def fit(self, ids: List[str], texts: List[str]):
        self.ids = ids
        if texts:
            self.mat = self.tfidf.fit_transform(texts)
        else:
            self.mat = None

    def get_candidates(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        if self.mat is None:
            return []
        qv = self.tfidf.transform([query])
        scores = (self.mat @ qv.T).toarray().squeeze()
        idx = np.argsort(-scores)[:top_k]
        results = []
        for i in idx:
            if scores[i] <= 0:
                continue
            results.append((self.ids[i], float(scores[i])))
        return results
