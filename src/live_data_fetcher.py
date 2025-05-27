"""
Windows-Robust LiveDataManager for Project Crypto Bot Peter
Specifically designed to handle Windows asyncio threading issues
"""

# Configure logging with more detail
import logging
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG for this testing phase
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

import asyncio
import aiohttp
import orjson # Make sure 'orjson' is in requirements.txt
import threading
import time
import ssl
import certifi # Make sure 'certifi' is in requirements.txt
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any
import sys
import numpy as np # Added for RingBuffer
import gc

# Windows-specific imports (optional, for ProactorEventLoop if winloop is used)
_use_winloop = False
if sys.platform == 'win32':
    try:
        import winloop # Make sure 'winloop' is in requirements.txt if we want to use it
        _use_winloop = True
        # winloop.install() # Call this once globally if using winloop for all asyncio
    except ImportError:
        logger.info("winloop not found, will attempt ProactorEventLoop or default asyncio loop.")
        pass


class TokenBucket:
    """Thread-safe token bucket rate limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity) # Ensure float for calculations
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.last_refill = time.monotonic() # Use monotonic clock for intervals
        self._lock = threading.Lock() # Use standard Lock, RLock not strictly needed here
    
    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                logger.debug(f"TokenBucket: Consumed {tokens}. Tokens left: {self.tokens:.2f}")
                return True
            logger.debug(f"TokenBucket: Consume failed. Tokens needed: {tokens}, available: {self.tokens:.2f}")
            return False


@dataclass
class DataFeedState:
    """Lightweight state tracking for live data feed"""
    last_successful_fetch: float = 0.0
    consecutive_errors: int = 0
    connection_healthy: bool = True
    total_fetches: int = 0
    
    def is_stale(self, max_age_seconds: int = 10) -> bool:
        return time.monotonic() - self.last_successful_fetch > max_age_seconds
    
    def record_success(self):
        self.last_successful_fetch = time.monotonic()
        self.consecutive_errors = 0
        self.connection_healthy = True
        self.total_fetches += 1
    
    def record_error(self):
        self.consecutive_errors += 1
        if self.consecutive_errors > 5: # Consider this threshold configurable
            self.connection_healthy = False
            logger.warning("DataFeedState: Connection marked unhealthy due to excessive consecutive errors.")


@dataclass
class PriceVolumeUpdate:
    price: float
    volume: float
    timestamp: float  # seconds since epoch
    latency_spike_flag: int = 0
    fetch_latency_ms: Optional[float] = None
    latency_volatility_index: Optional[np.float16] = None


class RingBuffer:
    """Memory-efficient ring buffer for price history"""
    
    def __init__(self, size: int = 1000):
        self._price = np.zeros(size, dtype=np.float32)
        self._volume = np.zeros(size, dtype=np.float32)
        self._timestamps = np.zeros(size, dtype=np.float64)
        self._index = 0
        self._size = size
        self._count = 0
        self._lock = threading.RLock()
        
    def append(self, price: float, volume: float, timestamp: float):
        with self._lock:
            idx = self._index % self._size
            self._price[idx] = price
            self._volume[idx] = volume
            self._timestamps[idx] = timestamp
            self._index += 1
            self._count = min(self._count + 1, self._size)
    
    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if n > self._count:
                n = self._count
            
            if n == 0:
                return (np.array([], dtype=np.float32),
                        np.array([], dtype=np.float32),
                        np.array([], dtype=np.float64))
            
            # Calculate indices correctly for ring buffer
            # Newest item is at (self._index - 1) % self._size
            # Oldest of the 'n' recent items is at (self._index - n) % self._size
            
            indices = [(self._index - n + i) % self._size for i in range(n)]
            
            return (self._price[indices].copy(),
                    self._volume[indices].copy(),
                    self._timestamps[indices].copy())
    
    @property
    def count(self):
        with self._lock:
            return self._count


class WindowsRobustLiveDataManager:
    """
    Windows-specific robust implementation of live data manager
    Handles Windows threading + asyncio + aiohttp issues
    """
    
    BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    BINANCE_TRADES_URL = "https://api.binance.com/api/v3/trades?symbol={symbol}&limit=1"
    BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/{symbol_lower}@trade"
    
    def __init__(self, symbol: str = "BTCUSDT", fetch_interval: float = 2.0):
        self.symbol = symbol
        self.symbol_lower = symbol.lower()
        self.fetch_interval = fetch_interval
        self.base_fetch_interval = fetch_interval
        self._latest: Optional[PriceVolumeUpdate] = None
        self._price_lock = threading.RLock()
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._fetch_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._rate_limiter = TokenBucket(capacity=15, refill_rate=0.25)
        self._state = DataFeedState()
        self._error_tracker = deque(maxlen=20)
        self._update_callback: Optional[Callable[[PriceVolumeUpdate], None]] = None
        self._buffer = RingBuffer(size=1000)
        self._mode = "websocket"  # or "http"
        self._recent_latencies = deque(maxlen=20)
        self._recent_latency_ms = deque(maxlen=20)  # For rolling stddev
        logger.info(f"WindowsRobustLiveDataManager initialized for {self.symbol}")
    
    def set_update_callback(self, callback: Callable[[PriceVolumeUpdate], None]):
        self._update_callback = callback
    
    def get_latest(self) -> Optional[PriceVolumeUpdate]:
        with self._price_lock:
            return self._latest
    
    def get_state(self) -> DataFeedState:
        return self._state

    def get_recent_for_fsm(self, n: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Provides price data for FSM, potentially from its own buffer"""
        return self._buffer.get_recent(n)

    def _create_windows_ssl_context(self) -> Optional[ssl.SSLContext]:
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            ssl_context.check_hostname = True # Default is True, but explicit
            ssl_context.verify_mode = ssl.CERT_REQUIRED # Default, but explicit
            # More robust cipher list as suggested by Claude, good for some restrictive environments
            ssl_context.set_ciphers('ECDHE+AESGCM:CHACHA20') # Simplified, modern list
            logger.debug("Custom SSL context created successfully using certifi.")
            return ssl_context
        except Exception as e:
            logger.warning(f"Failed to create custom SSL context with certifi: {e}. Falling back to default aiohttp SSL.")
            return None # Let aiohttp use its default

    async def _create_robust_session(self) -> aiohttp.ClientSession:
        ssl_context = self._create_windows_ssl_context()
        
        connector = aiohttp.TCPConnector(
            limit=1, # Single connection as per Claude's minimal example
            limit_per_host=1,
            ttl_dns_cache=60, 
            use_dns_cache=True, # CRITICAL per Claude
            keepalive_timeout=10,
            enable_cleanup_closed=True,
            ssl=ssl_context, # Pass our custom context
            resolver=aiohttp.AsyncResolver(), 
            family=0 
        )
        
        timeout = aiohttp.ClientTimeout(
            total=10, connect=5, sock_read=5, sock_connect=5
        )
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'CryptoBotPeter/1.1 (WindowsRobust)', 'Accept': 'application/json'},
            json_serialize=orjson.dumps, # Use orjson
            trust_env=False # Don't use system proxy settings, per Claude
        )
        logger.debug("Robust aiohttp session created.")
        return session
    
    async def _test_connectivity(self, session: aiohttp.ClientSession) -> bool:
        logger.info("WindowsRobustLiveDataManager: Testing connectivity...")
        url_to_test = self.BINANCE_TICKER_URL.format(symbol=self.symbol)
        try:
            async with session.get(url_to_test) as response:
                logger.info(f"WindowsRobustLiveDataManager: Connectivity test response status: HTTP {response.status}")
                if response.status == 200:
                    data = await response.read() # Read limited bytes
                    logger.info(f"WindowsRobustLiveDataManager: Test response size: {len(data)} bytes. Content (first 100): {data[:100]}")
                    parsed = orjson.loads(data)
                    price = float(parsed.get('price', 0))
                    logger.info(f"WindowsRobustLiveDataManager: Test price parsed: ${price:,.2f}. Connectivity OK.")
                    return True
                else:
                    logger.error(f"WindowsRobustLiveDataManager: Connectivity test failed with status {response.status}. Response: {await response.text()[:500]}")
                    return False
        except Exception as e:
            logger.error(f"WindowsRobustLiveDataManager: Connectivity test exception: {e}", exc_info=True)
            return False

    async def _fetch_http_price_volume(self, session: aiohttp.ClientSession) -> Optional[PriceVolumeUpdate]:
        if not self._rate_limiter.consume():
            logger.debug("Rate limit protection, skipping fetch.")
            return None
        try:
            start = time.perf_counter()
            # Price
            url_price = self.BINANCE_TICKER_URL.format(symbol=self.symbol)
            async with session.get(url_price) as resp:
                if resp.status != 200:
                    logger.warning(f"HTTP price fetch failed: {resp.status}")
                    self._state.record_error()
                    return None
                data = orjson.loads(await resp.read())
                price = float(data["price"])
            # Volume
            url_trades = self.BINANCE_TRADES_URL.format(symbol=self.symbol)
            async with session.get(url_trades) as resp:
                if resp.status != 200:
                    logger.warning(f"HTTP trades fetch failed: {resp.status}")
                    self._state.record_error()
                    return None
                data = orjson.loads(await resp.read())
                if not data:
                    logger.warning("No trades data returned.")
                    self._state.record_error()
                    return None
                trade = data[0]
                volume = float(trade["qty"])
                timestamp = trade["time"] / 1000.0  # ms to s
            latency = time.perf_counter() - start
            latency_ms = latency * 1000.0
            self._recent_latencies.append(latency)
            self._recent_latency_ms.append(latency_ms)
            median_latency = np.median(self._recent_latencies) if self._recent_latencies else 0.0
            latency_spike_flag = int(latency > 2 * median_latency and median_latency > 0)
            # Latency volatility index (stddev of ms)
            if len(self._recent_latency_ms) >= 2:
                lvi = np.std(self._recent_latency_ms)
                lvi = np.float16(lvi)
            else:
                lvi = np.float16(0.0)
            self._state.record_success()
            return PriceVolumeUpdate(price=price, volume=volume, timestamp=timestamp, latency_spike_flag=latency_spike_flag, fetch_latency_ms=latency_ms, latency_volatility_index=lvi)
        except Exception as e:
            logger.error(f"HTTP price/volume fetch error: {e}", exc_info=True)
            self._state.record_error()
            return None

    async def _websocket_loop(self, session: aiohttp.ClientSession):
        ws_url = self.BINANCE_WS_URL.format(symbol_lower=self.symbol_lower)
        logger.info(f"Connecting to Binance WebSocket: {ws_url}")
        try:
            async with session.ws_connect(ws_url, heartbeat=30) as ws:
                logger.info("WebSocket connection established.")
                self._mode = "websocket"
                async for msg in ws:
                    if not self._running:
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            ws_start = time.perf_counter()
                            data = orjson.loads(msg.data)
                            price = float(data["p"])
                            volume = float(data["q"])
                            timestamp = data["T"] / 1000.0
                            latency = time.perf_counter() - ws_start
                            latency_ms = latency * 1000.0
                            self._recent_latencies.append(latency)
                            self._recent_latency_ms.append(latency_ms)
                            median_latency = np.median(self._recent_latencies) if self._recent_latencies else 0.0
                            latency_spike_flag = int(latency > 2 * median_latency and median_latency > 0)
                            if len(self._recent_latency_ms) >= 2:
                                lvi = np.std(self._recent_latency_ms)
                                lvi = np.float16(lvi)
                            else:
                                lvi = np.float16(0.0)
                            update = PriceVolumeUpdate(price=price, volume=volume, timestamp=timestamp, latency_spike_flag=latency_spike_flag, fetch_latency_ms=latency_ms, latency_volatility_index=lvi)
                            with self._price_lock:
                                self._latest = update
                            self._buffer.append(price, volume, timestamp)
                            if self._update_callback:
                                try:
                                    self._update_callback(update)
                                except Exception as cb_error:
                                    logger.error(f"Update callback error: {cb_error}", exc_info=True)
                            self._state.record_success()
                        except Exception as parse_error:
                            logger.error(f"WebSocket message parse error: {parse_error}", exc_info=True)
                            self._state.record_error()
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {msg.data}")
                        self._state.record_error()
                        break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                        logger.warning("WebSocket closed by server.")
                        break
        except Exception as ws_error:
            logger.error(f"WebSocket connection error: {ws_error}", exc_info=True)
            self._state.record_error()
            raise

    async def _main_fetch_loop(self):
        logger.info("Main async fetch loop initiated.")
        self._session = await self._create_robust_session()
        logger.info("Robust session created for fetch loop.")
        try:
            # Try WebSocket first
            try:
                await self._websocket_loop(self._session)
            except Exception as ws_fail:
                logger.error(f"WebSocket failed, switching to HTTP polling: {ws_fail}")
                self._mode = "http"
                while self._running:
                    update = await self._fetch_http_price_volume(self._session)
                    if update:
                        with self._price_lock:
                            self._latest = update
                        self._buffer.append(update.price, update.volume, update.timestamp)
                        if self._update_callback:
                            try:
                                self._update_callback(update)
                            except Exception as cb_error:
                                logger.error(f"Update callback error: {cb_error}", exc_info=True)
                        self._state.record_success()
                    await asyncio.sleep(self.fetch_interval)
        finally:
            if self._session and not self._session.closed:
                logger.info("_main_fetch_loop finally - Closing aiohttp session...")
                try:
                    await self._session.close()
                    logger.info("_main_fetch_loop finally - Session closed successfully.")
                except Exception as close_error:
                    logger.error(f"_main_fetch_loop finally - Error closing session: {close_error}", exc_info=True)
            logger.info("_main_fetch_loop has finished.")

    def _setup_windows_event_loop(self):
        logger.debug("WindowsRobustLiveDataManager: Setting up event loop for thread.")
        if sys.platform == 'win32':
            if _use_winloop: # If winloop was successfully imported
                try:
                    # winloop.install() # Call this once globally if deciding to use winloop for all asyncio
                    self._loop = asyncio.ProactorEventLoop() # Or winloop.new_event_loop()
                    logger.info("WindowsRobustLiveDataManager: Using ProactorEventLoop (via winloop or asyncio).")
                except Exception as e_winloop:
                    logger.warning(f"WindowsRobustLiveDataManager: Failed to use ProactorEventLoop/winloop ({e_winloop}), falling back to default SelectorEventLoop.")
                    self._loop = asyncio.SelectorEventLoop()
            else: # winloop not available
                try: # Python 3.8+ on Windows, Proactor is available but Selector is default.
                    self._loop = asyncio.ProactorEventLoop()
                    logger.info("WindowsRobustLiveDataManager: Using asyncio.ProactorEventLoop for Windows.")
                except RuntimeError: # Proactor not supported in some contexts e.g. running inside another loop
                     logger.warning("WindowsRobustLiveDataManager: ProactorEventLoop not supported, using default asyncio loop.")
                     self._loop = asyncio.new_event_loop() # Fallback
        else: # Non-windows
            self._loop = asyncio.new_event_loop()
            logger.info("WindowsRobustLiveDataManager: Using default asyncio event loop (non-Windows).")
        
        asyncio.set_event_loop(self._loop)
        self._loop.set_debug(True) # Enable asyncio debug mode
        return self._loop

    def _run_event_loop_thread(self):
        logger.info(f"WindowsRobustLiveDataManager: Starting event loop thread: {threading.current_thread().name}")
        try:
            loop = self._setup_windows_event_loop()
            self._fetch_task = loop.create_task(self._main_fetch_loop())
            loop.run_until_complete(self._fetch_task)
        except Exception as e:
            logger.error(f"WindowsRobustLiveDataManager: Event loop thread encountered fatal error: {e}", exc_info=True)
        finally:
            logger.info("WindowsRobustLiveDataManager: Event loop thread exiting.")
            if self._loop and not self._loop.is_closed():
                try:
                    # Ensure all tasks are cancelled before stopping loop
                    for task in asyncio.all_tasks(self._loop):
                        if not task.done():
                            task.cancel()
                    # Run one last time to process cancellations
                    self._loop.run_until_complete(asyncio.sleep(0)) 
                    self._loop.close()
                    logger.info("WindowsRobustLiveDataManager: Event loop closed.")
                except Exception as close_error:
                    logger.error(f"WindowsRobustLiveDataManager: Error closing event loop: {close_error}", exc_info=True)
    
    def start(self):
        if self._running:
            logger.warning("WindowsRobustLiveDataManager: Already running.")
            return
        
        logger.info("WindowsRobustLiveDataManager: Starting...")
        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop_thread,
            name="LiveDataWorkerThread", # More descriptive name
            daemon=True
        )
        self._thread.start()
        logger.info("WindowsRobustLiveDataManager: Live data worker thread started.")
        time.sleep(0.1) # Give thread a moment to initialize loop and session

    def stop(self):
        if not self._running and not (self._thread and self._thread.is_alive()):
            logger.info("WindowsRobustLiveDataManager: Already stopped or not started.")
            return
        
        logger.info("WindowsRobustLiveDataManager: stop() called. Initiating shutdown...")
        self._running = False # Signal loops to stop
        
        if self._loop and self._fetch_task:
            if not self._loop.is_closed():
                logger.debug("WindowsRobustLiveDataManager: Calling loop.call_soon_threadsafe to cancel _fetch_task.")
                self._loop.call_soon_threadsafe(self._fetch_task.cancel)
            else:
                logger.warning("WindowsRobustLiveDataManager: Event loop already closed, cannot cancel _fetch_task via call_soon_threadsafe.")

        if self._thread and self._thread.is_alive():
            logger.debug(f"WindowsRobustLiveDataManager: Waiting for thread {self._thread.name} to join...")
            self._thread.join(timeout=10) # Increased timeout for graceful shutdown
            if self._thread.is_alive():
                logger.warning(f"WindowsRobustLiveDataManager: Thread {self._thread.name} did not stop cleanly after 10s.")
            else:
                logger.info(f"WindowsRobustLiveDataManager: Thread {self._thread.name} stopped successfully.")
        else:
            logger.info("WindowsRobustLiveDataManager: Worker thread already stopped or not started.")
        
        # Final check on event loop, moved from _run_event_loop_thread's finally
        if self._loop and not self._loop.is_closed():
            logger.debug("WindowsRobustLiveDataManager: Attempting final event loop stop and close from stop().")
            try:
                # Ensure all tasks are really cancelled
                async def cancel_all_tasks(loop):
                    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
                    if not tasks: return
                    logger.debug(f"Cancelling {len(tasks)} outstanding tasks in event loop.")
                    for task in tasks: task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    logger.debug("Outstanding tasks cancelled.")

                if not self._loop.is_running(): # If loop isn't running, can't run_until_complete
                     self._loop.call_soon_threadsafe(self._loop.stop)
                else:
                    future = asyncio.run_coroutine_threadsafe(cancel_all_tasks(self._loop), self._loop)
                    try:
                        future.result(timeout=5) # Wait for cancellation to complete
                    except asyncio.TimeoutError:
                        logger.warning("WindowsRobustLiveDataManager: Timeout waiting for tasks to cancel during stop.")
                    except Exception as e:
                        logger.error(f"WindowsRobustLiveDataManager: Error during task cancellation in stop: {e}")
                    self._loop.call_soon_threadsafe(self._loop.stop)
                
                # Give loop time to stop if it was running
                time.sleep(0.1) 
                if not self._loop.is_closed(): self._loop.close()
                logger.info("WindowsRobustLiveDataManager: Event loop definitively closed from stop().")
            except Exception as e_loop_close:
                logger.error(f"WindowsRobustLiveDataManager: Exception during final loop close in stop(): {e_loop_close}")

        logger.info("WindowsRobustLiveDataManager: Shutdown process complete.")

    def get_debug_info(self) -> dict:
        latest = self.get_latest()
        price_val = latest.price if latest else None
        price_ts = latest.timestamp if latest else None
        
        return {
            'running': self._running,
            'thread_alive': self._thread.is_alive() if self._thread else False,
            'latest_price_value': price_val,
            'latest_price_timestamp': price_ts,
            'price_age_seconds': time.monotonic() - price_ts if price_ts else None,
            'state': self._state, # Includes consecutive_errors, total_fetches, connection_healthy
            'error_tracker_count': len(self._error_tracker),
            'recent_errors_sample': list(self._error_tracker)[-3:] if self._error_tracker else [],
            'latency_spike_flag': latest.latency_spike_flag if latest else None
        }

def check_settings_dict(settings_dict, required_keys, dict_name):
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        logger.error(f"CRITICAL: Missing keys in {dict_name}: {missing}")
        from src.data_logger import log_event
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        raise RuntimeError(f"CRITICAL: Missing keys in {dict_name}: {missing}")

# Test harness (example usage)
if __name__ == "__main__":
    # This allows testing this module standalone
    def main_test():
        
        # Setup signal handling for graceful shutdown (Ctrl+C)
        manager_instance = None # To be accessible in signal_handler

        def graceful_signal_handler(sig, frame):
            print("\nSIGINT received, shutting down LiveDataManager gracefully...")
            if manager_instance:
                manager_instance.stop()
            # Allow some time for threads to clean up before forceful exit if needed
            # For a standalone test, we might just exit, but in an app, you'd manage this.
            sys.exit(0) 
        
        signal.signal(signal.SIGINT, graceful_signal_handler)
        signal.signal(signal.SIGTERM, graceful_signal_handler)

        # Create manager
        manager_instance = WindowsRobustLiveDataManager(symbol="BTCUSDT", fetch_interval=2.0) # Test with 2s interval

        # Set up callback to print prices
        def price_update_printer_callback(price: float, timestamp: float):
            fetch_time_monotonic = timestamp # Assuming LiveDataManager passes monotonic time
            age = time.monotonic() - fetch_time_monotonic
            print(f"📈 CALLBACK: New Price Received: ${price:,.2f} (Timestamp: {timestamp:.2f}, Age: {age:.2f}s)")
        
        manager_instance.set_price_callback(price_update_printer_callback)
        
        print("🚀 Starting WindowsRobustLiveDataManager for standalone test...")
        manager_instance.start()
        print("✅ WindowsRobustLiveDataManager started. Monitoring price updates...")
        print("Press Ctrl+C to stop.")
        
        try:
            # Keep main thread alive to observe background thread, print status
            counter = 0
            while True:
                time.sleep(5) # Print status every 5 seconds
                counter += 1
                debug_info = manager_instance.get_debug_info()
                price_val = debug_info['latest_price_value']
                price_age = debug_info['price_age_seconds']
                
                if price_val is not None and price_age is not None:
                    print(f"[{counter*5:3d}s elapsed] Main Thread Poll: Latest Price ${price_val:,.2f} ({price_age:.1f}s ago). "
                          f"State: Healthy={debug_info['state'].connection_healthy}, "
                          f"ConsecErrors={debug_info['state'].consecutive_errors}, "
                          f"TotalFetches={debug_info['state'].total_fetches}. "
                          f"ThreadAlive={debug_info['thread_alive']}")
                else:
                    print(f"[{counter*5:3d}s elapsed] Main Thread Poll: No price data yet. "
                          f"State: Healthy={debug_info['state'].connection_healthy}, "
                          f"ConsecErrors={debug_info['state'].consecutive_errors}, "
                          f"TotalFetches={debug_info['state'].total_fetches}. "
                          f"ThreadAlive={debug_info['thread_alive']}")
                
                if not debug_info['thread_alive'] and debug_info['running']:
                    logger.error("Main Thread Poll: Worker thread is not alive but manager is still in 'running' state!")
                    break
                if not debug_info['running'] and not debug_info['thread_alive']:
                    logger.info("Main Thread Poll: Manager and worker thread have stopped.")
                    break
        
        except KeyboardInterrupt:
            logger.info("Main Thread: KeyboardInterrupt received by main thread.")
        finally:
            logger.info("Main Thread: Initiating final stop sequence for LiveDataManager...")
            if manager_instance: # Ensure it exists
                manager_instance.stop()
            logger.info("Main Thread: Standalone test finished.")

    main_test() 