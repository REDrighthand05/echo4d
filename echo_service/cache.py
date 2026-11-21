"""Throttled filesystem caching to eliminate polling disk I/O churn."""

import time
from typing import Any, Callable, Optional


class ThrottledCache:
    """Thread-safe time-based cache for expensive filesystem scans."""

    def __init__(self, ttl_seconds: float = 2.5):
        self.ttl = ttl_seconds
        self._cache: Any = None
        self._cache_time: float = 0.0

    def get_or_scan(self, scanner: Callable[[], Any], force: bool = False) -> Any:
        """Return cached value if within TTL, otherwise invoke scanner and cache."""
        now = time.time()
        if not force and self._cache is not None and (now - self._cache_time < self.ttl):
            return self._cache
        try:
            data = scanner()
            self._cache = data
            self._cache_time = now
            return data
        except Exception:
            return self._cache

    def invalidate(self) -> None:
        """Invalidate the cache so the next call forces a fresh scan."""
        self._cache_time = 0.0

    @property
    def value(self) -> Any:
        return self._cache
