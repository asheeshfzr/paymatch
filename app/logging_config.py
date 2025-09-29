# app/logging_config.py
"""
Structured logging configuration with request IDs and correlation.
"""

import json
import logging
import sys
import uuid
from typing import Any, Dict, Optional
from contextvars import ContextVar
from datetime import datetime

from app.config import settings

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
transaction_id_var: ContextVar[Optional[str]] = ContextVar('transaction_id', default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add request context if available
        if settings.logging.include_request_id and request_id_var.get():
            log_entry['request_id'] = request_id_var.get()
        
        if settings.logging.include_correlation_id and correlation_id_var.get():
            log_entry['correlation_id'] = correlation_id_var.get()
        
        if user_id_var.get():
            log_entry['user_id'] = user_id_var.get()
        
        if transaction_id_var.get():
            log_entry['transaction_id'] = transaction_id_var.get()
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from the log record
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage',
                'exc_info', 'exc_text', 'stack_info'
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter with request context."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with request context."""
        # Add request context to message
        context_parts = []
        if settings.logging.include_request_id and request_id_var.get():
            context_parts.append(f"req_id={request_id_var.get()}")
        if settings.logging.include_correlation_id and correlation_id_var.get():
            context_parts.append(f"corr_id={correlation_id_var.get()}")
        if user_id_var.get():
            context_parts.append(f"user_id={user_id_var.get()}")
        if transaction_id_var.get():
            context_parts.append(f"txn_id={transaction_id_var.get()}")
        
        if context_parts:
            record.msg = f"[{' '.join(context_parts)}] {record.msg}"
        
        return super().format(record)


def setup_logging() -> None:
    """Setup application logging configuration."""
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.logging.level))
    
    # Set formatter based on configuration
    if settings.logging.format.lower() == 'json':
        formatter = StructuredFormatter()
    else:
        formatter = TextFormatter()
    
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(getattr(logging, settings.logging.level))
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info("Logging configured", extra={
        'level': settings.logging.level,
        'format': settings.logging.format,
        'include_request_id': settings.logging.include_request_id,
        'include_correlation_id': settings.logging.include_correlation_id
    })


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def set_request_context(
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    transaction_id: Optional[str] = None
) -> None:
    """Set request context variables."""
    if request_id:
        request_id_var.set(request_id)
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if user_id:
        user_id_var.set(user_id)
    if transaction_id:
        transaction_id_var.set(transaction_id)


def clear_request_context() -> None:
    """Clear request context variables."""
    request_id_var.set(None)
    correlation_id_var.set(None)
    user_id_var.set(None)
    transaction_id_var.set(None)


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


# Context manager for request logging
class RequestLoggingContext:
    """Context manager for request-scoped logging."""
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        transaction_id: Optional[str] = None
    ):
        self.request_id = request_id or generate_request_id()
        self.correlation_id = correlation_id or generate_correlation_id()
        self.user_id = user_id
        self.transaction_id = transaction_id
        self.old_request_id = None
        self.old_correlation_id = None
        self.old_user_id = None
        self.old_transaction_id = None
    
    def __enter__(self):
        # Store old values
        self.old_request_id = request_id_var.get()
        self.old_correlation_id = correlation_id_var.get()
        self.old_user_id = user_id_var.get()
        self.old_transaction_id = transaction_id_var.get()
        
        # Set new values
        set_request_context(
            request_id=self.request_id,
            correlation_id=self.correlation_id,
            user_id=self.user_id,
            transaction_id=self.transaction_id
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old values
        request_id_var.set(self.old_request_id)
        correlation_id_var.set(self.old_correlation_id)
        user_id_var.set(self.old_user_id)
        transaction_id_var.set(self.old_transaction_id)


# Initialize logging on import
setup_logging()
