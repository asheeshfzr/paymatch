# app/metrics.py
"""
Prometheus metrics for monitoring LLM calls, latency, token usage, and system performance.
"""

import time
from typing import Optional, Dict, Any
from functools import wraps

# Try to import Prometheus client
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    from prometheus_client.exposition import CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when Prometheus is not available
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
    
    Counter = Histogram = Gauge = DummyMetric
    Info = DummyMetric
    CollectorRegistry = DummyMetric
    generate_latest = lambda registry: b"# Prometheus metrics not available\n"
    CONTENT_TYPE_LATEST = "text/plain"

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Create a custom registry for our metrics
if PROMETHEUS_AVAILABLE:
    registry = CollectorRegistry()
else:
    registry = None

# LLM Metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['model', 'operation', 'status'],
    registry=registry
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['model', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens used in LLM requests',
    ['model', 'operation', 'token_type'],
    registry=registry
)

llm_cost_total = Counter(
    'llm_cost_total',
    'Total cost of LLM requests in USD',
    ['model', 'operation'],
    registry=registry
)

llm_retries_total = Counter(
    'llm_retries_total',
    'Total number of LLM retries',
    ['model', 'operation'],
    registry=registry
)

# API Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=registry
)

# Search and Matching Metrics
search_requests_total = Counter(
    'search_requests_total',
    'Total number of search requests',
    ['search_type', 'use_llm_expansion', 'use_llm_rerank', 'use_qdrant'],
    registry=registry
)

match_requests_total = Counter(
    'match_requests_total',
    'Total number of match requests',
    ['use_llm_parse'],
    registry=registry
)

candidates_found_total = Histogram(
    'candidates_found_total',
    'Number of candidates found in search/matching',
    ['operation_type'],
    buckets=[1, 5, 10, 20, 50, 100, 200, 500],
    registry=registry
)

# Embedding Metrics
embedding_requests_total = Counter(
    'embedding_requests_total',
    'Total number of embedding requests',
    ['operation_type'],
    registry=registry
)

embedding_duration_seconds = Histogram(
    'embedding_duration_seconds',
    'Embedding computation duration in seconds',
    ['operation_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Qdrant Metrics
qdrant_requests_total = Counter(
    'qdrant_requests_total',
    'Total number of Qdrant requests',
    ['operation', 'status'],
    registry=registry
)

qdrant_duration_seconds = Histogram(
    'qdrant_duration_seconds',
    'Qdrant request duration in seconds',
    ['operation'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

# System Metrics
active_requests = Gauge(
    'active_requests',
    'Number of active requests',
    registry=registry
)

# Application Info
app_info = Info(
    'app_info',
    'Application information',
    registry=registry
)

# Set application info
app_info.info({
    'name': settings.app_name,
    'version': settings.app_version,
    'environment': settings.environment
})


def record_llm_call(
    model: str,
    operation: str,
    duration: float,
    tokens_used: Optional[int] = None,
    cost: Optional[float] = None,
    success: bool = True,
    retries: int = 0
) -> None:
    """Record LLM call metrics."""
    status = 'success' if success else 'error'
    
    llm_requests_total.labels(model=model, operation=operation, status=status).inc()
    llm_request_duration_seconds.labels(model=model, operation=operation).observe(duration)
    
    if retries > 0:
        llm_retries_total.labels(model=model, operation=operation).inc(retries)
    
    if tokens_used:
        llm_tokens_total.labels(model=model, operation=operation, token_type='total').inc(tokens_used)
    
    if cost:
        llm_cost_total.labels(model=model, operation=operation).inc(cost)


def record_api_call(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float
) -> None:
    """Record API call metrics."""
    api_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)


def record_search_call(
    search_type: str,
    use_llm_expansion: bool,
    use_llm_rerank: bool,
    use_qdrant: bool,
    candidates_found: int
) -> None:
    """Record search call metrics."""
    search_requests_total.labels(
        search_type=search_type,
        use_llm_expansion=str(use_llm_expansion),
        use_llm_rerank=str(use_llm_rerank),
        use_qdrant=str(use_qdrant)
    ).inc()
    candidates_found_total.labels(operation_type=search_type).observe(candidates_found)


def record_match_call(
    use_llm_parse: bool,
    candidates_found: int
) -> None:
    """Record match call metrics."""
    match_requests_total.labels(use_llm_parse=str(use_llm_parse)).inc()
    candidates_found_total.labels(operation_type='match').observe(candidates_found)


def record_embedding_call(
    operation_type: str,
    duration: float
) -> None:
    """Record embedding call metrics."""
    embedding_requests_total.labels(operation_type=operation_type).inc()
    embedding_duration_seconds.labels(operation_type=operation_type).observe(duration)


def record_qdrant_call(
    operation: str,
    duration: float,
    success: bool = True
) -> None:
    """Record Qdrant call metrics."""
    status = 'success' if success else 'error'
    qdrant_requests_total.labels(operation=operation, status=status).inc()
    qdrant_duration_seconds.labels(operation=operation).observe(duration)


def record_cache_hit(cache_type: str) -> None:
    """Record cache hit."""
    cache_hits_total.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    """Record cache miss."""
    cache_misses_total.labels(cache_type=cache_type).inc()


def track_active_requests():
    """Decorator to track active requests."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            active_requests.inc()
            try:
                return await func(*args, **kwargs)
            finally:
                active_requests.dec()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            active_requests.inc()
            try:
                return func(*args, **kwargs)
            finally:
                active_requests.dec()
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def get_metrics() -> str:
    """Get Prometheus metrics in text format."""
    if not PROMETHEUS_AVAILABLE:
        return "# Prometheus metrics not available - install prometheus-client"
    return generate_latest(registry).decode('utf-8')


def get_metrics_content_type() -> str:
    """Get Prometheus metrics content type."""
    return CONTENT_TYPE_LATEST


# Token cost estimation (rough estimates for common models)
TOKEN_COSTS = {
    'gpt-4o-mini': {'input': 0.00015 / 1000, 'output': 0.0006 / 1000},
    'gpt-4o': {'input': 0.005 / 1000, 'output': 0.015 / 1000},
    'gpt-3.5-turbo': {'input': 0.0015 / 1000, 'output': 0.002 / 1000},
}


def estimate_token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for token usage."""
    if model not in TOKEN_COSTS:
        logger.warning(f"Unknown model for cost estimation: {model}")
        return 0.0
    
    costs = TOKEN_COSTS[model]
    return (input_tokens * costs['input']) + (output_tokens * costs['output'])


# Import asyncio for the decorator
import asyncio
