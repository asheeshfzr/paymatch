# app/qdrant_wrapper.py
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

class QdrantWrapper:
    def __init__(self, collection_name: str = "transactions", host: str = "localhost", port: int = 6333):
        self.collection = collection_name
        self.client = None
        self.exists = False
        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(host=host, port=port)
                self.exists = True
            except Exception:
                self.client = None
                self.exists = False

    def recreate_collection(self, vector_size: int, distance: str = "Cosine"):
        """Recreate Qdrant collection with error handling."""
        if not self.client:
            print("ERROR: Qdrant client not available")
            return False
        
        if vector_size <= 0:
            print(f"ERROR: Invalid vector size: {vector_size}")
            return False
        
        try:
            params = rest_models.VectorParams(
                size=vector_size, 
                distance=rest_models.Distance.COSINE if distance.lower()=="cosine" else rest_models.Distance.DOT
            )
            self.client.recreate_collection(collection_name=self.collection, vectors_config=params)
            print(f"INFO: Successfully recreated Qdrant collection '{self.collection}' with vector size {vector_size}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to recreate Qdrant collection: {e}")
            return False

    def upsert(self, ids: List[str], vectors: List[np.ndarray], payloads: Optional[List[Dict[str,Any]]] = None):
        """Upsert vectors to Qdrant collection with error handling."""
        if not self.client:
            print("ERROR: Qdrant client not available")
            return False
        
        if not ids or not vectors:
            print("ERROR: Empty ids or vectors provided to upsert")
            return False
        
        if len(ids) != len(vectors):
            print(f"ERROR: Mismatch between ids ({len(ids)}) and vectors ({len(vectors)}) length")
            return False
        
        try:
            points = []
            for idx, vec in enumerate(vectors):
                try:
                    payload = payloads[idx] if payloads and idx < len(payloads) else None
                    points.append(rest_models.PointStruct(id=ids[idx], vector=vec.tolist(), payload=payload))
                except Exception as e:
                    print(f"WARNING: Failed to create point for id {ids[idx]}: {e}")
                    continue
            
            if not points:
                print("ERROR: No valid points to upsert")
                return False
            
            self.client.upsert(collection_name=self.collection, points=points)
            print(f"INFO: Successfully upserted {len(points)} points to Qdrant collection '{self.collection}'")
            return True
        except Exception as e:
            print(f"ERROR: Failed to upsert to Qdrant collection: {e}")
            return False

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str,Any]]:
        """Search Qdrant collection with error handling."""
        if not self.client:
            print("ERROR: Qdrant client not available")
            return []
        
        if not query_vector:
            print("ERROR: Empty query vector provided")
            return []
        
        if top_k <= 0:
            print(f"WARNING: Invalid top_k value: {top_k}, using 10")
            top_k = 10
        
        try:
            hits = self.client.search(collection_name=self.collection, query_vector=query_vector, limit=top_k)
            results = []
            for h in hits:
                try:
                    results.append({"id": str(h.id), "score": float(h.score), "payload": h.payload})
                except Exception as e:
                    print(f"WARNING: Failed to process search hit: {e}")
                    continue
            return results
        except Exception as e:
            print(f"ERROR: Qdrant search failed: {e}")
            return []
