import pytest
import time
from experiments.utils.network_utils import RateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_wait_time():
    max_requests = 5
    window_seconds = 2
    rate_limiter = RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    # Simulate `max_requests` within the window
    start_time = time.time()
    for _ in range(max_requests):
        await rate_limiter.acquire()
    elapsed_time = time.time() - start_time

    # Ensure no wait occurred for the first `max_requests`
    assert elapsed_time < 0.1, f"Unexpected wait time: {elapsed_time:.2f}s"

    # Trigger the rate limit
    start_time = time.time()
    await rate_limiter.acquire()  # This should wait for the window to reset
    elapsed_time = time.time() - start_time

    # Ensure the wait time is approximately `window_seconds`
    assert elapsed_time == pytest.approx(window_seconds, abs=0.2), \
        f"Wait time was {elapsed_time:.2f}s, expected ~{window_seconds}s"