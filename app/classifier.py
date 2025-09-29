# app/classifier.py
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression

class MatchClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.trained = False

    def train(self, X: List[List[float]], y: List[int]):
        if len(X) < 10:
            # don't fail: fit trivial model if small data
            X_arr = np.array(X)
            y_arr = np.array(y) if len(y)>0 else np.zeros(len(X_arr))
            if len(X_arr.shape) == 1:
                X_arr = X_arr.reshape(-1,1)
            try:
                self.model.fit(X_arr, y_arr)
                self.trained = True
            except Exception:
                self.trained = False
            return
        X_arr = np.array(X)
        y_arr = np.array(y)
        self.model.fit(X_arr, y_arr)
        self.trained = True

    def predict_proba(self, X: List[List[float]]):
        X_arr = np.array(X)
        if not self.trained:
            # heuristic normalization
            s = X_arr.sum(axis=1)
            minv, maxv = s.min(), s.max()
            if maxv == minv:
                return [0.5 for _ in s]
            return ((s - minv) / (maxv - minv)).tolist()
        return self.model.predict_proba(X_arr)[:,1].tolist()
