# app/main_enhanced.py
"""
Enhanced FastAPI application with configuration, observability, caching, and async support.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import pandas as pd
import numpy as np

# Import our enhanced modules
from app.config import settings
from app.logging_config import get_logger, RequestLoggingContext, generate_request_id
from app.metrics import (
    record_api_call, record_search_call, record_match_call, 
    get_metrics, get_metrics_content_type, track_active_requests
)
from app.tracing import get_trace_context, set_trace_context
from app.cache import cache_manager, cached, search_cache_key, match_cache_key

# Import existing modules
from app.trigram_index import SimpleTrigramIndex
from app.bm25_prefilter import BM25Prefilter
from app.embedder import Embedder
from app.qdrant_wrapper import QdrantWrapper
from app.classifier import MatchClassifier
from app.matcher import init_components, hybrid_match
from app.parser import extract_candidate_names
from app.llm_utils_enhanced import llm_expand_query, llm_rerank, LLM_ENABLED

logger = get_logger(__name__)

# Global data stores
USERS: List[Dict] = []
TXNS: Dict[str, Dict] = {}
TRIGRAM: Optional[SimpleTrigramIndex] = None
USERS_BY_ID: Dict[str, Dict] = {}
BM25: Optional[BM25Prefilter] = None
EMB: Optional[Embedder] = None
QWRAP: Optional[QdrantWrapper] = None
CLF: Optional[MatchClassifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting application", extra={
        'app_name': settings.app_name,
        'version': settings.app_version,
        'environment': settings.environment
    })
    
    try:
        await initialize_application()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    if QWRAP:
        try:
            # Close any connections
            pass
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def initialize_application():
    """Initialize application components."""
    global USERS, TXNS, TRIGRAM, USERS_BY_ID, BM25, EMB, QWRAP, CLF
    
    # Load data
    USERS = load_users(settings.database.users_csv_path)
    TXNS = load_transactions(settings.database.transactions_csv_path)
    
    logger.info("Data loaded", extra={
        'users_count': len(USERS),
        'transactions_count': len(TXNS)
    })
    
    # Initialize trigram index
    TRIGRAM = SimpleTrigramIndex()
    USERS_BY_ID = {}
    
    for user in USERS:
        uid = user.get('id') or user.get('user_id') or user.get('uid') or ""
        name = user.get('name') or ""
        if uid:
            USERS_BY_ID[uid] = user
            TRIGRAM.add(uid, name)
    
    logger.info("Trigram index built", extra={
        'users_indexed': len(USERS_BY_ID)
    })
    
    # Initialize BM25 prefilter
    TXN_IDS = list(TXNS.keys())
    TXN_TEXTS = [(TXNS[tid].get('description', '') or "") for tid in TXN_IDS]
    
    BM25 = BM25Prefilter()
    if TXN_IDS:
        BM25.fit(TXN_IDS, TXN_TEXTS)
    
    logger.info("BM25 prefilter initialized", extra={
        'transactions_indexed': len(TXN_IDS)
    })
    
    # Initialize embedder
    EMB = Embedder()
    if TXN_IDS:
        EMB.fit_corpus(TXN_IDS, TXN_TEXTS)
    
    logger.info("Embedder initialized", extra={
        'model': EMB.model_name if hasattr(EMB, 'model_name') else 'unknown'
    })
    
    # Initialize Qdrant (optional)
    if settings.features.enable_qdrant:
        try:
            QWRAP = QdrantWrapper(
                collection_name=settings.qdrant.collection_name,
                host=settings.qdrant.host,
                port=settings.qdrant.port
            )
            
            if QWRAP.exists and EMB.fitted and EMB.corpus_embs is not None:
                vector_size = EMB.corpus_embs.shape[1]
                ok = QWRAP.recreate_collection(vector_size=vector_size, distance="Cosine")
                if ok:
                    QWRAP.upsert(
                        TXN_IDS,
                        [emb for emb in EMB.corpus_embs],
                        payloads=[{"description": d} for d in TXN_TEXTS]
                    )
                    logger.info("Qdrant collection populated")
                else:
                    logger.warning("Failed to recreate Qdrant collection")
            else:
                logger.info("Qdrant not available or embeddings not ready")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            QWRAP = None
    
    # Initialize classifier
    CLF = MatchClassifier()
    
    # Initialize matcher components
    init_components(TRIGRAM, EMB, USERS_BY_ID)
    
    logger.info("All components initialized successfully")


def load_users(path: str) -> List[Dict]:
    """Load users from CSV file."""
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to load users from {path}: {e}")
        return []


def load_transactions(path: str) -> Dict[str, Dict]:
    """Load transactions from CSV file."""
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        return {r['id']: r for r in df.to_dict(orient='records')}
    except Exception as e:
        logger.error(f"Failed to load transactions from {path}: {e}")
        return {}


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enhanced hybrid matching and search API with LLM support",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Middleware for request logging and tracing."""
    request_id = generate_request_id()
    start_time = time.time()
    
    # Set request context
    with RequestLoggingContext(request_id=request_id):
        # Extract trace context from headers
        trace_headers = dict(request.headers)
        set_trace_context(trace_headers)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record API metrics
        record_api_call(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        logger.info("Request processed", extra={
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'duration': duration,
            'request_id': request_id
        })
        
        return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "features": {
            "llm_enabled": LLM_ENABLED,
            "qdrant_enabled": QWRAP is not None and QWRAP.exists,
            "caching_enabled": settings.features.enable_caching
        }
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if not settings.features.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return cache_manager.get_stats()


@app.get("/cache/clear")
async def clear_cache(cache_type: Optional[str] = None):
    """Clear cache(s)."""
    cache_manager.clear(cache_type)
    return {"message": f"Cache cleared: {cache_type or 'all'}"}


@track_active_requests
@app.get("/match_transaction/{transaction_id}")
async def api_match_transaction(
    transaction_id: str,
    top_k: int = Query(10, ge=1, le=100),
    min_score: float = Query(30.0, ge=0.0, le=100.0),
    use_llm_for_parse: bool = Query(False)
):
    """Enhanced transaction matching endpoint with caching."""
    # Validate parameters
    if top_k > settings.api.max_top_k:
        raise HTTPException(
            status_code=400,
            detail=f"top_k cannot exceed {settings.api.max_top_k}"
        )
    
    # Check cache first
    cache_key = match_cache_key(transaction_id, top_k, min_score, use_llm_for_parse)
    cached_result = cache_manager.get("search_results", cache_key)
    if cached_result:
        logger.info("Returning cached match result", extra={
            'transaction_id': transaction_id,
            'cache_key': cache_key[:16] + '...'
        })
        return cached_result
    
    # Get transaction
    txn = TXNS.get(transaction_id)
    if txn is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found"
        )
    
    logger.info("Processing match request", extra={
        'transaction_id': transaction_id,
        'use_llm_for_parse': use_llm_for_parse,
        'llm_available': LLM_ENABLED
    })
    
    # Perform matching
    results = hybrid_match(
        txn,
        top_k=top_k,
        min_score=min_score/100.0,
        use_llm=use_llm_for_parse
    )
    
    # Filter by min_score
    filtered_results = [
        r for r in results if r["match_metric"] >= min_score
    ]
    
    # Record metrics
    record_match_call(use_llm_for_parse, len(filtered_results))
    
    response_data = {
        "users": filtered_results,
        "total_number_of_matches": len(filtered_results)
    }
    
    # Cache result
    cache_manager.set("search_results", cache_key, response_data, ttl=1800)  # 30 minutes
    
    return response_data


@track_active_requests
@app.get("/search_transactions/")
async def api_search_transactions(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
    use_qdrant: bool = Query(True),
    use_llm_for_expansion: bool = Query(False),
    use_llm_for_rerank: bool = Query(False)
):
    """Enhanced transaction search endpoint with caching and async support."""
    # Validate parameters
    if len(query) > settings.api.max_query_length:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long (max {settings.api.max_query_length} characters)"
        )
    
    if top_k > settings.api.max_top_k:
        raise HTTPException(
            status_code=400,
            detail=f"top_k cannot exceed {settings.api.max_top_k}"
        )
    
    # Check cache first
    cache_key = search_cache_key(query, top_k, use_llm_for_expansion, use_llm_for_rerank, use_qdrant)
    cached_result = cache_manager.get("search_results", cache_key)
    if cached_result:
        logger.info("Returning cached search result", extra={
            'query': query[:50] + '...' if len(query) > 50 else query,
            'cache_key': cache_key[:16] + '...'
        })
        return cached_result
    
    logger.info("Processing search request", extra={
        'query': query[:50] + '...' if len(query) > 50 else query,
        'use_llm_expansion': use_llm_for_expansion,
        'use_llm_rerank': use_llm_for_rerank,
        'use_qdrant': use_qdrant
    })
    
    # Count tokens
    token_count = EMB.count_tokens(query)
    
    # Step 1: Query expansion (LLM optional)
    queries = [query]
    if use_llm_for_expansion and LLM_ENABLED:
        try:
            queries = await asyncio.to_thread(llm_expand_query, query)
        except Exception as e:
            logger.error(f"LLM query expansion failed: {e}")
            queries = [query]
    
    # Step 2: BM25 candidate prefilter
    candidate_ids_set = set()
    for q in queries:
        cand_pairs = BM25.get_candidates(q, top_k=200)
        candidate_ids_set.update([cid for cid, _ in cand_pairs])
    
    candidate_ids = list(candidate_ids_set) if candidate_ids_set else list(TXNS.keys())
    
    # Step 3: Vector search
    q_emb = EMB.embed_query(query)
    
    if use_qdrant and QWRAP and QWRAP.exists:
        # Use Qdrant for vector search
        try:
            hits = await asyncio.to_thread(QWRAP.search, q_emb.tolist(), top_k=top_k)
            transactions = [
                {"id": h["id"], "embedding": float(h["score"])}
                for h in hits
            ]
            
            # Record metrics
            record_search_call("qdrant", use_llm_for_expansion, use_llm_for_rerank, use_qdrant, len(transactions))
            
            response_data = {
                "transactions": transactions,
                "total_number_of_tokens_used": int(token_count)
            }
            
            # Cache result
            cache_manager.set("search_results", cache_key, response_data, ttl=1800)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            # Fall back to local search
    
    # Fallback: Local vector search
    cand_texts = [TXNS[cid].get('description', '') for cid in candidate_ids]
    if not cand_texts:
        return {"transactions": [], "total_number_of_tokens_used": int(token_count)}
    
    # Batch embed candidates
    cand_embs = await asyncio.to_thread(EMB.embed_texts, cand_texts)
    sims = [float(np.dot(q_emb, e)) for e in cand_embs]
    
    # Step 4: Optional LLM reranking
    order = list(np.argsort(-np.array(sims)))
    if use_llm_for_rerank and LLM_ENABLED:
        try:
            rerank_idx = await asyncio.to_thread(llm_rerank, query, cand_texts)
            order = [i for i in rerank_idx if 0 <= i < len(cand_texts)]
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
    
    # Prepare results
    results = []
    for idx in order[:top_k]:
        results.append({
            "id": candidate_ids[idx],
            "embedding": sims[idx]
        })
    
    # Record metrics
    record_search_call("local", use_llm_for_expansion, use_llm_for_rerank, use_qdrant, len(results))
    
    response_data = {
        "transactions": results,
        "total_number_of_tokens_used": int(token_count)
    }
    
    # Cache result
    cache_manager.set("search_results", cache_key, response_data, ttl=1800)
    
    return response_data


# Expose lowercase alias for ASGI servers
main_app = app
