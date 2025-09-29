# app/cache.py
"""
Caching system for LLM responses, embeddings, and search results.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from functools import wraps
import asyncio

from app.config import settings
from app.logging_config import get_logger
from app.metrics import record_cache_hit, record_cache_miss

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = 0.0


class LRUCache:
    """Simple LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is expired."""
        if entry.expires_at is None:
            return False
        return time.time() > entry.expires_at
    
    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self.access_order:
            return
        
        lru_key = self.access_order[0]
        if lru_key in self.cache:
            del self.cache[lru_key]
            self.access_order.remove(lru_key)
    
    def _update_access(self, key: str) -> None:
        """Update access tracking for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if self._is_expired(entry):
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return None
        
        # Update access tracking
        entry.access_count += 1
        entry.last_accessed = time.time()
        self._update_access(key)
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in cache."""
        now = time.time()
        expires_at = None
        
        if ttl is not None:
            expires_at = now + ttl
        elif self.default_ttl is not None:
            expires_at = now + self.default_ttl
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=expires_at,
            access_count=1,
            last_accessed=now
        )
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = entry
        self._update_access(key)
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries,
            'max_size': self.max_size,
            'hit_rate': self._calculate_hit_rate() if hasattr(self, '_hits') else 0.0
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not hasattr(self, '_hits') or not hasattr(self, '_misses'):
            return 0.0
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class CacheManager:
    """Centralized cache manager for different cache types."""
    
    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}
        self.enabled = settings.features.enable_caching
        
        if self.enabled:
            # Initialize different cache types
            self.caches = {
                'llm_responses': LRUCache(max_size=500, default_ttl=3600),  # 1 hour
                'embeddings': LRUCache(max_size=1000, default_ttl=7200),    # 2 hours
                'search_results': LRUCache(max_size=200, default_ttl=1800), # 30 minutes
                'user_embeddings': LRUCache(max_size=1000, default_ttl=86400), # 24 hours
            }
            logger.info("Cache manager initialized", extra={
                'cache_types': list(self.caches.keys()),
                'enabled': self.enabled
            })
        else:
            logger.info("Caching disabled")
    
    def get_cache(self, cache_type: str) -> Optional[LRUCache]:
        """Get a specific cache instance."""
        if not self.enabled:
            return None
        return self.caches.get(cache_type)
    
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Get a value from a specific cache."""
        cache = self.get_cache(cache_type)
        if not cache:
            record_cache_miss(cache_type)
            return None
        
        value = cache.get(key)
        if value is not None:
            record_cache_hit(cache_type)
        else:
            record_cache_miss(cache_type)
        
        return value
    
    def set(self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in a specific cache."""
        cache = self.get_cache(cache_type)
        if cache:
            cache.set(key, value, ttl)
    
    def delete(self, cache_type: str, key: str) -> bool:
        """Delete a key from a specific cache."""
        cache = self.get_cache(cache_type)
        if cache:
            return cache.delete(key)
        return False
    
    def clear(self, cache_type: Optional[str] = None) -> None:
        """Clear cache(s)."""
        if cache_type:
            cache = self.get_cache(cache_type)
            if cache:
                cache.clear()
        else:
            for cache in self.caches.values():
                cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        if not self.enabled:
            return {'enabled': False}
        
        stats = {'enabled': True, 'caches': {}}
        for cache_type, cache in self.caches.items():
            stats['caches'][cache_type] = cache.stats()
        
        return stats


# Global cache manager
cache_manager = CacheManager()


def cached(
    cache_type: str,
    ttl: Optional[float] = None,
    key_func: Optional[callable] = None
):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not cache_manager.enabled:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.get_cache(cache_type)._generate_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_type, cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}", extra={
                    'cache_type': cache_type,
                    'cache_key': cache_key[:16] + '...'
                })
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_type, cache_key, result, ttl)
            
            logger.debug(f"Cache miss for {func.__name__}, stored result", extra={
                'cache_type': cache_type,
                'cache_key': cache_key[:16] + '...'
            })
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not cache_manager.enabled:
                return await func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.get_cache(cache_type)._generate_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_type, cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}", extra={
                    'cache_type': cache_type,
                    'cache_key': cache_key[:16] + '...'
                })
                return cached_result
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_type, cache_key, result, ttl)
            
            logger.debug(f"Cache miss for {func.__name__}, stored result", extra={
                'cache_type': cache_type,
                'cache_key': cache_key[:16] + '...'
            })
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Specialized cache key functions
def llm_cache_key(operation: str, *args, **kwargs) -> str:
    """Generate cache key for LLM operations."""
    key_data = {
        'operation': operation,
        'args': args,
        'kwargs': sorted(kwargs.items()) if kwargs else {},
        'model': settings.llm.model
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def embedding_cache_key(text: str, model: str) -> str:
    """Generate cache key for embeddings."""
    key_data = {
        'text': text,
        'model': model
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def search_cache_key(query: str, top_k: int, use_llm_expansion: bool, use_llm_rerank: bool, use_qdrant: bool) -> str:
    """Generate cache key for search operations."""
    key_data = {
        'query': query,
        'top_k': top_k,
        'use_llm_expansion': use_llm_expansion,
        'use_llm_rerank': use_llm_rerank,
        'use_qdrant': use_qdrant
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def match_cache_key(transaction_id: str, top_k: int, min_score: float, use_llm_parse: bool) -> str:
    """Generate cache key for match operations."""
    key_data = {
        'transaction_id': transaction_id,
        'top_k': top_k,
        'min_score': min_score,
        'use_llm_parse': use_llm_parse
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()
