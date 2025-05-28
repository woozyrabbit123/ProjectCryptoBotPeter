import psutil
import gc
import logging
from psutil import Process # Added import

logger = logging.getLogger(__name__)

class MemoryWatchdog:
    process: Process
    baseline_rss_mb: float
    growth_threshold_mb: int
    check_interval_fetches: int
    fetch_counter: int

    def __init__(self, growth_threshold_mb: int = 100, check_interval_fetches: int = 100):
        self.process: Process = psutil.Process() # Typed process attribute
        self.growth_threshold_mb: int = growth_threshold_mb
        self.check_interval_fetches: int = check_interval_fetches
        self.fetch_counter: int = 0
        self.baseline_rss_mb: float = self._get_current_rss_mb() # Refactored
        logger.info(f"MemoryWatchdog initialized. Baseline RSS: {self.baseline_rss_mb:.2f} MB")

    def _get_current_rss_mb(self) -> float: # Helper method
        return self.process.memory_info().rss / (1024 * 1024)

    def periodic_check(self) -> None: # Return type None
        self.fetch_counter += 1
        if self.fetch_counter >= self.check_interval_fetches:
            self.fetch_counter = 0
            current_rss_mb: float = self._get_current_rss_mb() # Refactored
            growth_mb: float = current_rss_mb - self.baseline_rss_mb
            logger.debug(f"MemoryWatchdog: Current RSS: {current_rss_mb:.2f} MB, Growth: {growth_mb:.2f} MB")
            if growth_mb > self.growth_threshold_mb:
                logger.warning(f"MemoryWatchdog: RSS growth ({growth_mb:.2f}MB) exceeded threshold ({self.growth_threshold_mb}MB). Forcing gc.collect().")
                gc.collect()
                self.baseline_rss_mb = self._get_current_rss_mb() # Refactored
                logger.info(f"MemoryWatchdog: New baseline RSS after GC: {self.baseline_rss_mb:.2f} MB") 