import psutil
import gc
import logging

logger = logging.getLogger(__name__)

class MemoryWatchdog:
    def __init__(self, growth_threshold_mb: int = 100, check_interval_fetches: int = 100):
        self.process = psutil.Process()
        self.baseline_rss_mb = self.process.memory_info().rss / (1024 * 1024)
        self.growth_threshold_mb = growth_threshold_mb
        self.check_interval_fetches = check_interval_fetches
        self.fetch_counter = 0
        logger.info(f"MemoryWatchdog initialized. Baseline RSS: {self.baseline_rss_mb:.2f} MB")

    def periodic_check(self):
        self.fetch_counter += 1
        if self.fetch_counter >= self.check_interval_fetches:
            self.fetch_counter = 0
            current_rss_mb = self.process.memory_info().rss / (1024 * 1024)
            growth_mb = current_rss_mb - self.baseline_rss_mb
            logger.debug(f"MemoryWatchdog: Current RSS: {current_rss_mb:.2f} MB, Growth: {growth_mb:.2f} MB")
            if growth_mb > self.growth_threshold_mb:
                logger.warning(f"MemoryWatchdog: RSS growth ({growth_mb:.2f}MB) exceeded threshold ({self.growth_threshold_mb}MB). Forcing gc.collect().")
                gc.collect()
                # Update baseline after collection to monitor new growth
                self.baseline_rss_mb = self.process.memory_info().rss / (1024 * 1024)
                logger.info(f"MemoryWatchdog: New baseline RSS after GC: {self.baseline_rss_mb:.2f} MB") 