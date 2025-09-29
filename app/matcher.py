# app/matcher.py
"""
Matcher logic for Task 1 - hybrid ensemble.
Supports optional LLM parsing when extracting candidate names.
"""

from typing import List, Dict, Any
from rapidfuzz import fuzz
import numpy as np

# These will be initialized by main.py via init_components
TRIGRAM_INDEX = None
EMBEDDER = None
USERS_BY_ID = {}
USER_EMBS = {}  # precomputed user embeddings


def init_components(trigram_index, embedder, users_by_id):
    """
    Initialize global components for matching.
    Precompute user embeddings to avoid repeated work per-request.
    """
    global TRIGRAM_INDEX, EMBEDDER, USERS_BY_ID, USER_EMBS
    TRIGRAM_INDEX = trigram_index
    EMBEDDER = embedder
    USERS_BY_ID = users_by_id

    # Precompute user embeddings (names)
    USER_EMBS = {}
    try:
        ids = list(users_by_id.keys())
        names = [users_by_id[uid].get("name", "") for uid in ids]
        if names:
            embs = EMBEDDER.embed_texts(names)
            USER_EMBS = {uid: emb for uid, emb in zip(ids, embs)}
    except Exception:
        USER_EMBS = {}


def _norm(s) -> str:
    return str(s).strip() if s else ""


def fuzzy_feats(a: str, b: str):
    a = _norm(a); b = _norm(b)
    if not a or not b:
        return [0.0, 0.0, 0.0]
    return [
        fuzz.ratio(a, b) / 100.0,
        fuzz.partial_ratio(a, b) / 100.0,
        fuzz.token_sort_ratio(a, b) / 100.0
    ]


def compute_score(candidate_name: str, user_rec: Dict[str, Any], txn_desc: str) -> float:
    """
    Compute a normalized score (0.0 - 1.0) for how well candidate_name matches user_rec,
    using fuzzy metrics, trigram hits, embedding similarity (name ↔ user name)
    and description ↔ user embedding similarity.
    """
    uname = user_rec.get("name", "")
    # fuzzy features
    f0, f1, f2 = fuzzy_feats(candidate_name, uname)

    # trigram hits
    tri_hits = 0
    try:
        tri_res = TRIGRAM_INDEX.query(candidate_name, top_k=200)
        for uid, cnt in tri_res:
            if uid == user_rec['id']:
                tri_hits = cnt
                break
    except Exception:
        tri_hits = 0
    tri_score = min(tri_hits / 10.0, 1.0)

    # embedding similarity: candidate name vs user name
    emb_sim = 0.0
    try:
        q_emb = EMBEDDER.embed_query(candidate_name)
        u_emb = USER_EMBS.get(user_rec['id'])
        if u_emb is not None:
            emb_sim = float(np.dot(q_emb, u_emb))
    except Exception:
        emb_sim = 0.0

    # description -> user embedding similarity
    desc_sim = 0.0
    try:
        d_emb = EMBEDDER.embed_query(txn_desc)
        u_emb2 = USER_EMBS.get(user_rec['id'])
        if u_emb2 is not None:
            desc_sim = float(np.dot(d_emb, u_emb2))
    except Exception:
        desc_sim = 0.0

    # Weighted ensemble (interpretable)
    # weights: fuzzy (avg of three metrics): 0.25, trigram:0.20, name-embed:0.30, desc-embed:0.25
    fuzzy_avg = (f0 + f1 + f2) / 3.0
    score = 0.25 * fuzzy_avg + 0.20 * tri_score + 0.30 * emb_sim + 0.25 * desc_sim
    return max(0.0, min(score, 1.0))


def hybrid_match(transaction: Dict[str, Any], top_k: int = 10, min_score: float = 0.3, use_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Main hybrid matching pipeline for a single transaction with comprehensive error handling.
    - Extract candidate payer names (optionally using LLM)
    - Use trigram index to get candidate users
    - Compute ensemble score per candidate-user pair
    - Keep best score per user and return top_k results
    """
    try:
        # Validate inputs
        if not transaction:
            raise ValueError("Transaction cannot be None or empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        desc = transaction.get("description", "") or ""
        if not desc:
            print("WARNING: Transaction has no description")
            return []
        
        # Import parser lazily (parser supports use_llm flag)
        try:
            from app.parser import extract_candidate_names
            candidates = extract_candidate_names(desc, max_candidates=3, use_llm=use_llm)
        except Exception as e:
            print(f"ERROR: Failed to extract candidate names: {e}")
            candidates = [desc]
        
        if not candidates:
            candidates = [desc]

        user_scores: Dict[str, float] = {}
        for cand in candidates:
            try:
                # get candidate user ids via trigram
                candidate_user_ids = []
                if TRIGRAM_INDEX:
                    try:
                        trig_res = TRIGRAM_INDEX.query(cand, top_k=200)
                        candidate_user_ids = [uid for uid, _ in trig_res]
                    except Exception as e:
                        print(f"WARNING: Trigram query failed for '{cand}': {e}")
                        candidate_user_ids = []
                else:
                    print("WARNING: Trigram index not available")

                if not candidate_user_ids:
                    candidate_user_ids = list(USERS_BY_ID.keys())[:200]

                for uid in candidate_user_ids:
                    try:
                        urec = USERS_BY_ID.get(uid)
                        if not urec:
                            continue
                        s = compute_score(cand, urec, desc)
                        prev = user_scores.get(uid, 0.0)
                        if s > prev:
                            user_scores[uid] = s
                    except Exception as e:
                        print(f"WARNING: Failed to compute score for user {uid}: {e}")
                        continue
                        
            except Exception as e:
                print(f"ERROR: Failed to process candidate '{cand}': {e}")
                continue

        # prepare results sorted by score (0..100)
        try:
            results = [{"id": uid, "match_metric": round(user_scores[uid] * 100.0, 2)} for uid in user_scores]
            results_sorted = sorted(results, key=lambda x: x["match_metric"], reverse=True)[:top_k]
            return results_sorted
        except Exception as e:
            print(f"ERROR: Failed to sort results: {e}")
            return []
            
    except Exception as e:
        print(f"ERROR: hybrid_match failed: {e}")
        return []
