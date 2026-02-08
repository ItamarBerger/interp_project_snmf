import asyncio
import time
import threading
from aiolimiter import AsyncLimiter
import logging
import re

logger = logging.getLogger(__name__)

EXPONENTIAL_BACKOFF_BASE = 2
BASE_SLEEP = 2


class RequestStats:
    """
    Thread-safe Singleton class to track API usage statistics.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RequestStats, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.total_requests = 0
        self.success = 0
        self.errors_429 = 0
        self.errors_other = 0
        self.start_time = time.time()
        self._initialized = True

    def reset(self):
        """Resets stats for a new run."""
        self.total_requests = 0
        self.success = 0
        self.errors_429 = 0
        self.errors_other = 0
        self.start_time = time.time()

    def log_status(self):
        """Logs the current success rate."""
        if self.total_requests == 0:
            return

        rate = (self.success / self.total_requests) * 100
        duration = time.time() - self.start_time
        rpm = (self.success / duration) * 60 if duration > 0 else 0

        logger.info(
            f"\n"
            f"========== Request Statistics ==========\n"
            f"Success Rate: {rate:.1f}% \n "
            f"Success RPM: {rpm:.0f} \n"
            f"429 errors: {self.errors_429} \n"
            f"Current rate of requests: {self.total_requests / duration:.1f} req/s \n"
            f"Successful requests out of total: {self.success}/{self.total_requests}\n"
            f"====================== =========="
        )

class RateLimiter:
    def __init__(self, max_requests: int = 25, window_seconds: int = 1):
        self._limiter = AsyncLimiter(max_rate=max_requests, time_period=window_seconds)
        self.total_requests = 0

    async def acquire(self):
        """Blocks until a slot is available in the current second."""
        async with self._limiter:
            self.total_requests += 1
            if self.total_requests % 100 == 0:
                logger.info(f"[Rate Limiter] Total requests passed: {self.total_requests}")
            return




def retry_with_attempts(attempts: int, default_value=0, stats: RequestStats = None):
    """
    Decorator to handle retries and exception handling for async functions.
    """
    if stats is None:
        stats = RequestStats()

    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                stats.total_requests += 1
                if stats.total_requests % 50 == 0:
                    stats.log_status()
                try:
                    result =  await func(*args, **kwargs)
                    if result is None:
                        if attempt == attempts - 1:
                            logger.warning(f"Function returned None, returning {default_value}.")
                            stats.errors_other += 1
                            return default_value
                    stats.success += 1
                    return result
                except asyncio.TimeoutError:
                    stats.errors_other += 1
                    logger.error(f"TIMEOUT: Attempt {attempt + 1} exceeded timeout")
                    if attempt == attempts - 1:
                        logger.warning("Skipping, returning %s.", default_value)
                        return default_value
                except Exception as e:
                    error_str = str(e)
                    retry_delay = 0
                    # Check for 429 error and extract retry_delay
                    if "429" in error_str or "quota" in error_str.lower():
                        stats.errors_429 += 1
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
                                f"Attempt {attempt + 1}: Got rate limit error: {error_str}, but no retry delay was specified, applying exponential backoff: {retry_delay:.1f}s...")
                            await asyncio.sleep(retry_delay)
                            continue

                    # Handle Standard Errors
                    stats.errors_other += 1
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == attempts - 1:
                        logger.warning("Skipping, returning %s.", default_value)
                        return default_value
        return wrapper
    return decorator