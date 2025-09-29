# app/main.py
from fastapi import FastAPI, HTTPException, Query
import logging
from pathlib import Path
import os, pandas as pd, numpy as np
from typing import Dict
from app.trigram_index import SimpleTrigramIndex
from app.bm25_prefilter import BM25Prefilter
from app.embedder import Embedder
from app.qdrant_wrapper import QdrantWrapper
from app.classifier import MatchClassifier
from app.matcher import init_components, hybrid_match
from app.parser import extract_candidate_names
from app.llm_utils import llm_expand_query, llm_rerank, LLM_ENABLED as LLM_AVAILABLE

# Ensure application logs at INFO level (so our module logs are visible)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP = FastAPI(title="Hybrid Optimal: Task1 + Task2", version="1.0")
# Expose lowercase alias for ASGI servers expecting module:variable as `app.main:app`
app = APP

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
USERS_CSV = os.getenv("USERS_CSV_PATH", str(DATA_DIR / "users.csv"))
TXN_CSV = os.getenv("TRANSACTIONS_CSV_PATH", str(DATA_DIR / "transactions.csv"))
if not Path(USERS_CSV).exists() and Path("/mnt/data/users.csv").exists():
    USERS_CSV = "/mnt/data/users.csv"
if not Path(TXN_CSV).exists() and Path("/mnt/data/transactions.csv").exists():
    TXN_CSV = "/mnt/data/transactions.csv"

def load_users(path):
    """Load users from CSV file with comprehensive error handling."""
    try:
        if not Path(path).exists():
            print(f"WARNING: Users CSV file not found: {path}")
            return []
        
        df = pd.read_csv(path, dtype=str).fillna("")
        if df.empty:
            print(f"WARNING: Users CSV file is empty: {path}")
            return []
        
        recs = df.to_dict(orient="records")
        print(f"INFO: Loaded {len(recs)} users from {path}")
        return recs
    except pd.errors.EmptyDataError:
        print(f"ERROR: Users CSV file is empty or corrupted: {path}")
        return []
    except pd.errors.ParserError as e:
        print(f"ERROR: Failed to parse users CSV file {path}: {e}")
        return []
    except FileNotFoundError:
        print(f"ERROR: Users CSV file not found: {path}")
        return []
    except PermissionError:
        print(f"ERROR: Permission denied reading users CSV file: {path}")
        return []
    except Exception as e:
        print(f"ERROR: Unexpected error loading users from {path}: {e}")
        return []

def load_txns(path):
    """Load transactions from CSV file with comprehensive error handling."""
    try:
        if not Path(path).exists():
            print(f"WARNING: Transactions CSV file not found: {path}")
            return {}
        
        df = pd.read_csv(path, dtype=str).fillna("")
        if df.empty:
            print(f"WARNING: Transactions CSV file is empty: {path}")
            return {}
        
        if 'id' not in df.columns:
            print(f"ERROR: Transactions CSV file missing 'id' column: {path}")
            return {}
        
        recs = {r['id']: r for r in df.to_dict(orient='records')}
        print(f"INFO: Loaded {len(recs)} transactions from {path}")
        return recs
    except pd.errors.EmptyDataError:
        print(f"ERROR: Transactions CSV file is empty or corrupted: {path}")
        return {}
    except pd.errors.ParserError as e:
        print(f"ERROR: Failed to parse transactions CSV file {path}: {e}")
        return {}
    except FileNotFoundError:
        print(f"ERROR: Transactions CSV file not found: {path}")
        return {}
    except PermissionError:
        print(f"ERROR: Permission denied reading transactions CSV file: {path}")
        return {}
    except KeyError as e:
        print(f"ERROR: Missing required column in transactions CSV file {path}: {e}")
        return {}
    except Exception as e:
        print(f"ERROR: Unexpected error loading transactions from {path}: {e}")
        return {}

USERS = load_users(USERS_CSV)
TXNS = load_txns(TXN_CSV)

# Build trigram index and user map
TRIGRAM = SimpleTrigramIndex()
USERS_BY_ID: Dict[str, Dict] = {}
try:
    for u in USERS:
        try:
            uid = u.get('id') or u.get('user_id') or u.get('uid') or ""
            name = u.get('name') or ""
            if not uid:
                continue
            USERS_BY_ID[uid] = u
            TRIGRAM.add(uid, name)
        except Exception as e:
            print(f"WARNING: Failed to process user record: {e}")
            continue
    print(f"INFO: Built trigram index with {len(USERS_BY_ID)} users")
except Exception as e:
    print(f"ERROR: Failed to build trigram index: {e}")
    USERS_BY_ID = {}

# BM25 prefilter for transaction corpus
TXN_IDS = list(TXNS.keys())
TXN_TEXTS = [ (TXNS[tid].get('description','') or "") for tid in TXN_IDS ]
BM25 = BM25Prefilter()
try:
    if TXN_IDS:
        BM25.fit(TXN_IDS, TXN_TEXTS)
        print(f"INFO: BM25 prefilter fitted with {len(TXN_IDS)} transactions")
    else:
        print("WARNING: No transactions available for BM25 prefilter")
except Exception as e:
    print(f"ERROR: Failed to fit BM25 prefilter: {e}")
    BM25 = None

# Embedder (Sentence-BERT preferred)
EMB = Embedder()
try:
    if TXN_IDS:
        EMB.fit_corpus(TXN_IDS, TXN_TEXTS)
        print(f"INFO: Embedder fitted with {len(TXN_IDS)} transactions")
    else:
        print("WARNING: No transactions available for embedder")
except Exception as e:
    print(f"ERROR: Failed to fit embedder: {e}")
    EMB = None

# Qdrant wrapper (optional)
QWRAP = QdrantWrapper(collection_name="transactions", host=os.getenv("QDRANT_HOST","localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))
if QWRAP.exists and EMB.fitted and EMB.corpus_embs is not None:
    try:
        vector_size = EMB.corpus_embs.shape[1]
        ok = QWRAP.recreate_collection(vector_size=vector_size, distance="Cosine")
        if ok:
            QWRAP.upsert(TXN_IDS, [emb for emb in EMB.corpus_embs], payloads=[{"description": d} for d in TXN_TEXTS])
    except Exception as e:
        pass

# Classifier training (kept for compatibility, not used by ensemble)
CLF = MatchClassifier()

# init matcher components (precompute user embeddings inside matcher)
init_components(TRIGRAM, EMB, USERS_BY_ID)

# --- Task 1 endpoint ---
@APP.get("/match_transaction/{transaction_id}")
def api_match_transaction(transaction_id: str,
                          top_k: int = Query(10, ge=1, le=100),
                          min_score: float = Query(30.0, ge=0.0, le=100.0),
                          use_llm_for_parse: bool = Query(False)):
    """Match a transaction to users with comprehensive error handling."""
    try:
        # Validate inputs
        if not transaction_id or not transaction_id.strip():
            raise HTTPException(status_code=400, detail="Transaction ID cannot be empty")
        
        if top_k <= 0 or top_k > 1000:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 1000")
        
        if min_score < 0 or min_score > 100:
            raise HTTPException(status_code=400, detail="min_score must be between 0 and 100")
        
        logger.info("Request: match_transaction id=%s use_llm_for_parse=%s LLM_AVAILABLE=%s", transaction_id, use_llm_for_parse, LLM_AVAILABLE)
        
        # Check if transaction exists
        txn = TXNS.get(transaction_id)
        if txn is None:
            raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
        
        # Check if required components are available
        if not USERS_BY_ID:
            raise HTTPException(status_code=503, detail="User data not available")
        
        # Call hybrid_match with error handling
        try:
            results = hybrid_match(txn, top_k=top_k, min_score=min_score/100.0, use_llm=use_llm_for_parse)
        except Exception as e:
            print(f"ERROR: hybrid_match failed for transaction {transaction_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal error during matching")
        
        # Filter results by min_score
        try:
            min_score_pct = float(min_score)
            filtered = [r for r in results if r.get("match_metric", 0) >= min_score_pct]
        except (ValueError, TypeError) as e:
            print(f"ERROR: Failed to filter results: {e}")
            filtered = results
        
        return {"users": filtered, "total_number_of_matches": len(filtered)}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error in match_transaction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Task 2 endpoint ---
@APP.get("/search_transactions/")
def api_search_transactions(query: str = Query(..., min_length=1),
                            top_k: int = Query(10, ge=1, le=100),
                            use_qdrant: bool = Query(True),
                            use_llm_for_expansion: bool = Query(False),
                            use_llm_for_rerank: bool = Query(False)):
    """Search transactions with comprehensive error handling."""
    try:
        # Validate inputs
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
        
        if top_k <= 0 or top_k > 1000:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 1000")
        
        # Check if required components are available
        if not TXNS:
            raise HTTPException(status_code=503, detail="Transaction data not available")
        
        if not EMB:
            raise HTTPException(status_code=503, detail="Embedder not available")
        
        # Count tokens with error handling
        try:
            token_count = EMB.count_tokens(query)
        except Exception as e:
            print(f"WARNING: Failed to count tokens: {e}")
            token_count = 0

        # Step 1: query expansion (LLM optional)
        queries = [query]
        if use_llm_for_expansion and LLM_AVAILABLE:
            try:
                queries = llm_expand_query(query)
            except Exception as e:
                print(f"WARNING: LLM query expansion failed: {e}")
                queries = [query]

        # Step 2: BM25 candidate prefilter (aggregate candidates from expansions)
        candidate_ids_set = set()
        try:
            if BM25:
                for q in queries:
                    try:
                        cand_pairs = BM25.get_candidates(q, top_k=200)
                        candidate_ids_set.update([cid for cid, _ in cand_pairs])
                    except Exception as e:
                        print(f"WARNING: BM25 search failed for query '{q}': {e}")
                        continue
            else:
                print("WARNING: BM25 not available, using all transactions")
        except Exception as e:
            print(f"ERROR: BM25 prefilter failed: {e}")
        
        candidate_ids = list(candidate_ids_set) if candidate_ids_set else TXN_IDS
    

        # Step 3: Qdrant or local vector search
        try:
            q_emb = EMB.embed_query(query)
        except Exception as e:
            print(f"ERROR: Failed to embed query: {e}")
            raise HTTPException(status_code=500, detail="Failed to process query")
        
        if use_qdrant and QWRAP and QWRAP.exists:
            try:
                hits = QWRAP.search(q_emb.tolist(), top_k=top_k)
                if hits:  # Only use Qdrant results if we have hits
                    transactions = [{"id": h["id"], "embedding": float(h["score"])} for h in hits]
                    return {"transactions": transactions, "total_number_of_tokens_used": int(token_count)}
                else:
                    # Qdrant returned no hits, fall back to local search
                    pass
            except Exception as e:
                print(f"WARNING: Qdrant search failed, falling back to local search: {e}")

        # Fallback: compute embeddings & similarity for candidate set
        try:
            cand_texts = [TXNS[cid].get('description','') for cid in candidate_ids]
            if not cand_texts:
                return {"transactions": [], "total_number_of_tokens_used": int(token_count)}
            
            cand_embs = EMB.embed_texts(cand_texts)
            sims = [float(np.dot(q_emb, e)) for e in cand_embs]
        except Exception as e:
            print(f"ERROR: Failed to compute embeddings: {e}")
            raise HTTPException(status_code=500, detail="Failed to compute similarity")

        # Step 4: optional LLM rerank (reorders candidate indices)
        try:
            order = list(np.argsort(-np.array(sims)))
            if use_llm_for_rerank and LLM_AVAILABLE:
                try:
                    # pass the candidate text snippets to LLM to get preferred order
                    rerank_idx = llm_rerank(query, cand_texts)
                    # rerank_idx is a list of indices (0-based). We'll use it but clip to available indices
                    # Keep only indices that are valid and in cand_texts
                    order = [i for i in rerank_idx if 0 <= i < len(cand_texts)]
                except Exception as e:
                    print(f"WARNING: LLM reranking failed: {e}")
                    order = list(np.argsort(-np.array(sims)))
        except Exception as e:
            print(f"ERROR: Failed to sort results: {e}")
            order = list(range(len(cand_texts)))

        results = []
        try:
            for idx in order[:top_k]:
                if idx < len(candidate_ids) and idx < len(sims):
                    results.append({"id": candidate_ids[idx], "embedding": sims[idx]})
        except Exception as e:
            print(f"ERROR: Failed to format results: {e}")
            results = []
        
        return {"transactions": results, "total_number_of_tokens_used": int(token_count)}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error in search_transactions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
