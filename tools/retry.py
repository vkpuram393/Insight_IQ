"""
Simple retry decorator.
Retries only when exception type is in `retry_on` AND exception.retriable is True.
"""

import functools
import time
import logging

# ============================================================================
# LOGGER
# ============================================================================
logger = logging.getLogger(__name__)

# ============================================================================
# RETRY DECORATOR
# ============================================================================
def retry(attempts: int = 3, retry_on: tuple = ()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    if attempt > 1:
                        logger.debug("Retry attempt %d for %s", attempt, func.__name__)
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    should_retry = isinstance(exc, retry_on) and getattr(exc, "retriable", False)
                    logger.debug("Exception on attempt %d: %s (retriable=%s)", attempt, exc, should_retry)
                    if not should_retry or attempt == attempts:
                        break
                    # backoff (simple)
                    time.sleep(0.5 * attempt)
            # exhausted or non-retriable -> re-raise last exception
            raise last_exc
        return wrapper
    return decorator
