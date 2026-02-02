import asyncio
from aiolimiter import AsyncLimiter
import logging
import re

logger = logging.getLogger(__name__)

EXPONENTIAL_BACKOFF_BASE = 2
BASE_SLEEP = 2

class RateLimiter:
    """
    Drop-in replacement that uses a Token Bucket (aiolimiter)
    to smooth traffic evenly over time, preventing micro-bursts.
    """
    def __init__(self, max_requests: int = 25, window_seconds: int = 1):
        # We default to 25 requests per 1 second (approx 1500 RPM).
        # This prevents the "spike" that triggers 429s.
        self._limiter = AsyncLimiter(max_rate=max_requests, time_period=window_seconds)
        self.total_requests = 0

    async def acquire(self):
        """Blocks until a slot is available in the current second."""
        async with self._limiter:
            self.total_requests += 1
            if self.total_requests % 100 == 0:
                logger.info(f"[Rate Limiter] Total requests passed: {self.total_requests}")
            return




def retry_with_attempts(attempts: int, default_value=0):
    """
    Decorator to handle retries and exception handling for async functions.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    result =  await func(*args, **kwargs)
                    if result is None:
                        if attempt == attempts - 1:
                            logger.warning("Function returned None, returning %s.", default_value)
                            return default_value
                    return result
                except asyncio.TimeoutError:
                    logger.error(f"TIMEOUT: Attempt {attempt + 1} exceeded timeout", exc_info=True)
                    if attempt == attempts - 1:
                        logger.warning("Skipping, returning %s.", default_value)
                        return default_value
                except Exception as e:
                    error_str = str(e)
                    retry_delay = 0
                    # Check for 429 error and extract retry_delay
                    if "429" in error_str or "quota" in error_str.lower():
                        retry_match = re.search(r'retry.*?(\d+\.?\d*)\s*s', error_str, re.IGNORECASE)
                        if retry_match:
                            retry_delay = float(retry_match.group(1))
                        elif "retry_delay" in error_str:
                            seconds_match = re.search(r'seconds:\s*(\d+)', error_str)
                            if seconds_match:
                                retry_delay = float(seconds_match.group(1))
                        if retry_delay > 0:
                            logger.info(f"Rate limit exceeded, waiting {retry_delay:.1f}s before retry...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            # Exponential backoff when no retry_delay is specified
                            retry_delay = BASE_SLEEP * (EXPONENTIAL_BACKOFF_BASE ** attempt)
                            logger.info(
                                f"Attempt {attempt + 1}: Got rate limit error, but no retry delay was specified, applying exponential backoff: {retry_delay:.1f}s...")
                            await asyncio.sleep(retry_delay)
                            continue
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == attempts - 1:
                        logger.warning("Skipping, returning %s.", default_value)
                        return default_value
        return wrapper
    return decorator