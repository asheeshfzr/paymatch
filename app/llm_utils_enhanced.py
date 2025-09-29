# app/llm_utils_enhanced.py
"""
Enhanced LLM utilities with timeouts, retries, response validation, and cost tracking.
"""

import asyncio
import time
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from functools import wraps
import backoff

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import settings
from app.logging_config import get_logger, RequestLoggingContext
from app.metrics import record_llm_call, estimate_token_cost
from app.tracing import trace_llm_call
from app.prompt_templates import format_prompt, validate_llm_response, get_llm_config

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    content: str
    model: str
    tokens_used: int
    cost: float
    duration: float
    retries: int
    success: bool
    error: Optional[str] = None


class LLMClient:
    """Enhanced LLM client with robustness features."""
    
    def __init__(self):
        self.sync_client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        self.enabled = settings.features.enable_llm and bool(settings.llm.api_key)
        
        if self.enabled:
            try:
                self.sync_client = OpenAI(
                    api_key=settings.llm.api_key,
                    timeout=settings.llm.timeout
                )
                self.async_client = AsyncOpenAI(
                    api_key=settings.llm.api_key,
                    timeout=settings.llm.timeout
                )
                logger.info("LLM client initialized successfully", extra={
                    'model': settings.llm.model,
                    'timeout': settings.llm.timeout,
                    'max_retries': settings.llm.max_retries
                })
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.enabled = False
                self.sync_client = None
                self.async_client = None
        else:
            logger.info("LLM client disabled (no API key or feature disabled)")
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        retryable_errors = (
            "timeout",
            "rate_limit",
            "server_error",
            "service_unavailable",
            "internal_server_error"
        )
        error_str = str(exception).lower()
        return any(err in error_str for err in retryable_errors)
    
    def _extract_token_usage(self, response: ChatCompletion) -> Tuple[int, float]:
        """Extract token usage and cost from response."""
        if not response.usage:
            return 0, 0.0
        
        total_tokens = response.usage.total_tokens
        cost = estimate_token_cost(
            settings.llm.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        
        return total_tokens, cost
    
    def _validate_response(self, operation: str, response: str) -> bool:
        """Validate LLM response format."""
        try:
            return validate_llm_response(operation, response)
        except Exception as e:
            logger.warning(f"Response validation failed for {operation}: {e}")
            return False
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=4,  # settings.llm.max_retries + 1
        giveup=lambda e: not self._should_retry(e),
        base=2,
        max_value=10
    )
    def _make_request(
        self,
        operation: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Make a synchronous LLM request with retries."""
        if not self.enabled or not self.sync_client:
            return LLMResponse(
                content="",
                model=settings.llm.model,
                tokens_used=0,
                cost=0.0,
                duration=0.0,
                retries=0,
                success=False,
                error="LLM not enabled or client unavailable"
            )
        
        start_time = time.time()
        retries = 0
        
        try:
            response = self.sync_client.chat.completions.create(
                model=settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=settings.llm.timeout
            )
            
            duration = time.time() - start_time
            
            if response and response.choices:
                content = response.choices[0].message.content or ""
                tokens_used, cost = self._extract_token_usage(response)
                
                # Validate response format
                is_valid = self._validate_response(operation, content)
                if not is_valid:
                    logger.warning(f"Invalid response format for {operation}: {content[:100]}...")
                
                # Check cost budget
                if cost > settings.llm.cost_budget_per_request:
                    logger.warning(f"LLM call exceeded cost budget: {cost:.4f} > {settings.llm.cost_budget_per_request}")
                
                result = LLMResponse(
                    content=content,
                    model=settings.llm.model,
                    tokens_used=tokens_used,
                    cost=cost,
                    duration=duration,
                    retries=retries,
                    success=True
                )
                
                # Record metrics
                record_llm_call(
                    model=settings.llm.model,
                    operation=operation,
                    duration=duration,
                    tokens_used=tokens_used,
                    cost=cost,
                    success=True,
                    retries=retries
                )
                
                logger.info(f"LLM call succeeded", extra={
                    'operation': operation,
                    'model': settings.llm.model,
                    'tokens_used': tokens_used,
                    'cost': cost,
                    'duration': duration,
                    'retries': retries
                })
                
                return result
            else:
                raise Exception("Empty response from LLM")
                
        except Exception as e:
            duration = time.time() - start_time
            retries += 1
            
            # Record metrics for failed call
            record_llm_call(
                model=settings.llm.model,
                operation=operation,
                duration=duration,
                success=False,
                retries=retries
            )
            
            logger.error(f"LLM call failed", extra={
                'operation': operation,
                'model': settings.llm.model,
                'error': str(e),
                'duration': duration,
                'retries': retries
            })
            
            if retries >= settings.llm.max_retries:
                return LLMResponse(
                    content="",
                    model=settings.llm.model,
                    tokens_used=0,
                    cost=0.0,
                    duration=duration,
                    retries=retries,
                    success=False,
                    error=str(e)
                )
            else:
                raise  # Let backoff handle retry
    
    async def _make_async_request(
        self,
        operation: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Make an asynchronous LLM request."""
        if not self.enabled or not self.async_client:
            return LLMResponse(
                content="",
                model=settings.llm.model,
                tokens_used=0,
                cost=0.0,
                duration=0.0,
                retries=0,
                success=False,
                error="LLM not enabled or client unavailable"
            )
        
        start_time = time.time()
        
        try:
            response = await self.async_client.chat.completions.create(
                model=settings.llm.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=settings.llm.timeout
            )
            
            duration = time.time() - start_time
            
            if response and response.choices:
                content = response.choices[0].message.content or ""
                tokens_used, cost = self._extract_token_usage(response)
                
                result = LLMResponse(
                    content=content,
                    model=settings.llm.model,
                    tokens_used=tokens_used,
                    cost=cost,
                    duration=duration,
                    retries=0,
                    success=True
                )
                
                # Record metrics
                record_llm_call(
                    model=settings.llm.model,
                    operation=operation,
                    duration=duration,
                    tokens_used=tokens_used,
                    cost=cost,
                    success=True
                )
                
                return result
            else:
                raise Exception("Empty response from LLM")
                
        except Exception as e:
            duration = time.time() - start_time
            
            record_llm_call(
                model=settings.llm.model,
                operation=operation,
                duration=duration,
                success=False
            )
            
            return LLMResponse(
                content="",
                model=settings.llm.model,
                tokens_used=0,
                cost=0.0,
                duration=duration,
                retries=0,
                success=False,
                error=str(e)
            )


# Global LLM client instance
llm_client = LLMClient()


@trace_llm_call("name_extraction")
def llm_extract_name(description: str) -> str:
    """
    Extract payer name from transaction description using LLM with validation.
    """
    if not description or not llm_client.enabled:
        return ""
    
    try:
        # Format prompt using template
        prompt = format_prompt("name_extraction", description=description)
        config = get_llm_config("name_extraction")
        
        # Make request
        response = llm_client._make_request(
            operation="name_extraction",
            prompt=prompt,
            max_tokens=config['max_tokens'],
            temperature=config['temperature']
        )
        
        if not response.success or not response.content:
            return ""
        
        # Clean and validate result
        content = response.content.strip()
        
        # Remove extra punctuation and normalize
        content = re.sub(r'[^A-Za-z\s]', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        if not content:
            return ""
        
        logger.info(f"LLM extracted name: '{content}'", extra={
            'original_description': description[:100],
            'extracted_name': content,
            'tokens_used': response.tokens_used,
            'cost': response.cost
        })
        
        return content
        
    except Exception as e:
        logger.error(f"LLM name extraction failed: {e}")
        return ""


@trace_llm_call("query_expansion")
def llm_expand_query(query: str) -> List[str]:
    """
    Use LLM to generate query paraphrases with validation.
    """
    if not query or not llm_client.enabled:
        return [query]
    
    try:
        # Format prompt using template
        prompt = format_prompt("query_expansion", query=query)
        config = get_llm_config("query_expansion")
        
        # Make request
        response = llm_client._make_request(
            operation="query_expansion",
            prompt=prompt,
            max_tokens=config['max_tokens'],
            temperature=config['temperature']
        )
        
        if not response.success or not response.content:
            return [query]
        
        # Parse response
        lines = [line.strip() for line in response.content.splitlines() if line.strip()]
        results = [query]  # Start with original query
        
        for line in lines:
            if line and line not in results:
                results.append(line)
                if len(results) >= 4:  # Original + 3 paraphrases
                    break
        
        logger.info(f"LLM expanded query to {len(results)} variants", extra={
            'original_query': query,
            'expanded_queries': results,
            'tokens_used': response.tokens_used,
            'cost': response.cost
        })
        
        return results
        
    except Exception as e:
        logger.error(f"LLM query expansion failed: {e}")
        return [query]


@trace_llm_call("reranking")
def llm_rerank(query: str, candidates: List[str]) -> List[int]:
    """
    Use LLM to rerank candidates with improved validation.
    """
    if not candidates or not llm_client.enabled:
        return list(range(len(candidates)))
    
    try:
        # Format candidates for prompt
        candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])
        
        # Format prompt using template
        prompt = format_prompt("reranking", query=query, candidates=candidates_text)
        config = get_llm_config("reranking")
        
        # Make request
        response = llm_client._make_request(
            operation="reranking",
            prompt=prompt,
            max_tokens=config['max_tokens'],
            temperature=config['temperature']
        )
        
        if not response.success or not response.content:
            return list(range(len(candidates)))
        
        # Parse and validate response
        content = response.content.strip()
        
        # Extract numbers from response
        numbers = re.findall(r'\d+', content)
        order = []
        
        for num_str in numbers:
            try:
                idx = int(num_str) - 1  # Convert to 0-based
                if 0 <= idx < len(candidates) and idx not in order:
                    order.append(idx)
            except ValueError:
                continue
        
        # Fill in missing indices
        for i in range(len(candidates)):
            if i not in order:
                order.append(i)
        
        logger.info(f"LLM reranked {len(candidates)} candidates", extra={
            'query': query,
            'rerank_order': order,
            'tokens_used': response.tokens_used,
            'cost': response.cost
        })
        
        return order
        
    except Exception as e:
        logger.error(f"LLM reranking failed: {e}")
        return list(range(len(candidates)))


# Async versions for better performance
async def llm_extract_name_async(description: str) -> str:
    """Async version of name extraction."""
    if not description or not llm_client.enabled:
        return ""
    
    try:
        prompt = format_prompt("name_extraction", description=description)
        config = get_llm_config("name_extraction")
        
        response = await llm_client._make_async_request(
            operation="name_extraction",
            prompt=prompt,
            max_tokens=config['max_tokens'],
            temperature=config['temperature']
        )
        
        if not response.success or not response.content:
            return ""
        
        content = response.content.strip()
        content = re.sub(r'[^A-Za-z\s]', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content if content else ""
        
    except Exception as e:
        logger.error(f"Async LLM name extraction failed: {e}")
        return ""


async def llm_expand_query_async(query: str) -> List[str]:
    """Async version of query expansion."""
    if not query or not llm_client.enabled:
        return [query]
    
    try:
        prompt = format_prompt("query_expansion", query=query)
        config = get_llm_config("query_expansion")
        
        response = await llm_client._make_async_request(
            operation="query_expansion",
            prompt=prompt,
            max_tokens=config['max_tokens'],
            temperature=config['temperature']
        )
        
        if not response.success or not response.content:
            return [query]
        
        lines = [line.strip() for line in response.content.splitlines() if line.strip()]
        results = [query]
        
        for line in lines:
            if line and line not in results:
                results.append(line)
                if len(results) >= 4:
                    break
        
        return results
        
    except Exception as e:
        logger.error(f"Async LLM query expansion failed: {e}")
        return [query]


# Backward compatibility
LLM_ENABLED = llm_client.enabled
