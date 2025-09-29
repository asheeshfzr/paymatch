# app/tracing.py
"""
OpenTelemetry tracing configuration for distributed tracing across API, embeddings, Qdrant, and LLM calls.
"""

import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# Try to import OpenTelemetry components
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx opentelemetry-instrumentation-openai opentelemetry-instrumentation-requests")


def setup_tracing() -> None:
    """Setup OpenTelemetry tracing."""
    if not OTEL_AVAILABLE or not settings.features.enable_tracing:
        logger.info("Tracing disabled or OpenTelemetry not available")
        return
    
    try:
        # Create resource
        resource = Resource.create({
            "service.name": settings.app_name,
            "service.version": settings.app_version,
            "service.namespace": "deel-staff-ai",
            "deployment.environment": settings.environment,
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Add OTLP exporter if configured
        otlp_endpoint = settings.tracing.get('otlp_endpoint') if hasattr(settings, 'tracing') else None
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
        
        # Auto-instrument libraries
        FastAPIInstrumentor.instrument_app()
        HTTPXClientInstrumentor().instrument()
        OpenAIInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        
        logger.info("OpenTelemetry tracing configured successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
        # Continue without tracing


def get_tracer(name: str):
    """Get a tracer instance."""
    if not OTEL_AVAILABLE:
        return None
    return trace.get_tracer(name)


def trace_function(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """Decorator to trace function calls."""
    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = get_tracer(func.__module__)
            if not tracer:
                return func(*args, **kwargs)
            
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function info
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return await func(*args, **kwargs)
            
            tracer = get_tracer(func.__module__)
            if not tracer:
                return await func(*args, **kwargs)
            
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function info
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    tracer_name: str = "app"
):
    """Context manager for creating spans."""
    if not OTEL_AVAILABLE:
        yield
        return
    
    tracer = get_tracer(tracer_name)
    if not tracer:
        yield
        return
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def trace_llm_call(
    model: str,
    operation: str,
    prompt_length: Optional[int] = None,
    response_length: Optional[int] = None
):
    """Trace LLM calls with specific attributes."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = get_tracer("llm")
            if not tracer:
                return func(*args, **kwargs)
            
            with tracer.start_as_current_span(f"llm.{operation}") as span:
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.operation", operation)
                
                if prompt_length:
                    span.set_attribute("llm.prompt_length", prompt_length)
                if response_length:
                    span.set_attribute("llm.response_length", response_length)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    span.set_attribute("llm.duration", duration)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("llm.duration", duration)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_embedding_call(operation_type: str):
    """Trace embedding calls."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = get_tracer("embedding")
            if not tracer:
                return func(*args, **kwargs)
            
            with tracer.start_as_current_span(f"embedding.{operation_type}") as span:
                span.set_attribute("embedding.operation", operation_type)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    span.set_attribute("embedding.duration", duration)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("embedding.duration", duration)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_qdrant_call(operation: str):
    """Trace Qdrant calls."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = get_tracer("qdrant")
            if not tracer:
                return func(*args, **kwargs)
            
            with tracer.start_as_current_span(f"qdrant.{operation}") as span:
                span.set_attribute("qdrant.operation", operation)
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    span.set_attribute("qdrant.duration", duration)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("qdrant.duration", duration)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def get_trace_context() -> Dict[str, str]:
    """Get current trace context for propagation."""
    if not OTEL_AVAILABLE:
        return {}
    
    propagator = TraceContextTextMapPropagator()
    carrier = {}
    propagator.inject(carrier)
    return carrier


def set_trace_context(carrier: Dict[str, str]) -> None:
    """Set trace context from carrier."""
    if not OTEL_AVAILABLE:
        return
    
    propagator = TraceContextTextMapPropagator()
    context = propagator.extract(carrier)
    trace.set_span_in_context(context)


# Import asyncio for the decorator
import asyncio

# Initialize tracing on import if enabled
if settings.features.enable_tracing:
    setup_tracing()
