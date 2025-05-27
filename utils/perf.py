import time
import logging
from functools import wraps
from typing import Callable, Any

_perf_enabled = False

def set_perf_enabled(enabled: bool) -> None:
    global _perf_enabled
    _perf_enabled = enabled

def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _perf_enabled:
            return func(*args, **kwargs)
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        delta = end - start
        logging.debug(f"PERF|{func.__module__}.{func.__name__}|{delta}ns")
        return result
    return wrapper 