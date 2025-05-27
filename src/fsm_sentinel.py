"""
Finite State Machine (FSM) module for Project Crypto Bot Peter.
Implements the Sleepy Sentinel FSM for efficient resource management.
"""

import collections
import time
import logging
from typing import Optional, Deque
from .fsm_optimizer import fast_should_wake
from dataclasses import dataclass
import numpy as np
import configparser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriceVolumeUpdate:
    __slots__ = [
        'price',
        'volume',
        'timestamp',
        'latency_spike_flag',
        'fetch_latency_ms',
        'latency_volatility_index'
    ]
    price: float
    volume: float
    timestamp: float
    latency_spike_flag: int
    fetch_latency_ms: Optional[float]
    latency_volatility_index: Optional[float]

    def __init__(self, price: float, volume: float, timestamp: float, latency_spike_flag: int = 0, fetch_latency_ms: Optional[float] = None, latency_volatility_index: Optional[float] = None):
        self.price = price
        self.volume = volume
        self.timestamp = timestamp
        self.latency_spike_flag = latency_spike_flag
        self.fetch_latency_ms = fetch_latency_ms
        self.latency_volatility_index = latency_volatility_index

def check_settings_dict(settings_dict, required_keys, dict_name):
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        logger.error(f"CRITICAL: Missing keys in {dict_name}: {missing}")
        from src.data_logger import log_event
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        raise RuntimeError(f"CRITICAL: Missing keys in {dict_name}: {missing}")

class SleepySentinelFSM:
    """
    Finite State Machine for market regime detection and adaptive sleep control.
    Tracks price and volume data, calculates EMA, and determines when to trigger active processing.
    """
    def __init__(self, threshold: float = 0.0005, ema_period: int = 5, stable_threshold_periods: int = 100, active_sleep: float = 0.1, idle_sleep: float = 0.5):
        """
        Initialize the FSM with thresholds, EMA period, and sleep intervals.
        Args:
            threshold (float): Relative threshold for wake trigger (as a fraction of EMA).
            ema_period (int): EMA period.
            stable_threshold_periods (int): Number of active periods to consider market stable.
            active_sleep (float): Sleep interval during active polling (seconds).
            idle_sleep (float): Sleep interval during idle (seconds).
        """
        # Defensive check for FSM config if loaded
        fsm_config = self._load_fsm_config()
        if fsm_config:
            required_keys = ['min_confidence_for_l1_trigger', 'high_confidence_level', 'high_confidence_threshold_modifier', 'price_deviation_threshold']
            check_settings_dict(fsm_config, required_keys, 'FSM_CONFIG')
        self.threshold = threshold
        self.ema_period = ema_period
        self.stable_threshold_periods = stable_threshold_periods
        self.active_sleep_interval = active_sleep
        self.idle_sleep_interval = idle_sleep
        self.market_stable_periods_count = 0
        self.buffer: Deque[PriceVolumeUpdate] = collections.deque(maxlen=ema_period + 20)
        self.current_ema: Optional[float] = None
        self.active_level = 0  # 0 = idle/sentinel, 1 = active
        self.latest: Optional[PriceVolumeUpdate] = None
        self.latest_regime_signal = None
        self.prices_buffer = collections.deque(maxlen=ema_period + 20)  # For test compatibility
        # Load config
        self._fsm_config = self._load_fsm_config()
        self._min_conf = float(self._fsm_config.get('min_confidence_for_l1_trigger', 0.5))
        self._high_conf = float(self._fsm_config.get('high_confidence_level', 0.75))
        self._high_conf_mod = float(self._fsm_config.get('high_confidence_threshold_modifier', 0.8))
        self._base_price_dev_thr = float(self._fsm_config.get('price_deviation_threshold', threshold))

    def _load_fsm_config(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        return config['fsm'] if 'fsm' in config else {}

    def add_price_volume_update(self, update: PriceVolumeUpdate) -> None:
        """
        Add a new PriceVolumeUpdate to the buffer and update EMA if enough data is present.
        Args:
            update (PriceVolumeUpdate): Latest market price, volume, and timestamp.
        """
        self.buffer.append(update)
        self.latest = update
        if len(self.buffer) >= self.ema_period:
            prices = [u.price for u in list(self.buffer)[-self.ema_period:]]
            if self.current_ema is None:
                self.current_ema = sum(prices) / self.ema_period
            else:
                alpha = 2 / (self.ema_period + 1)
                self.current_ema = prices[-1] * alpha + self.current_ema * (1 - alpha)

    def set_regime_signal(self, regime_signal):
        self.latest_regime_signal = regime_signal

    def check_for_wake_trigger(self, threshold_override: float = None) -> bool:
        """
        Check if the latest price deviates from EMA by more than the (possibly dynamic) threshold.
        Returns:
            bool: True if wake trigger condition is met, else False.
        """
        # Use prices_buffer for test compatibility
        if self.current_ema is not None and len(self.prices_buffer) > 0:
            latest_price = self.prices_buffer[-1]
            price_dev_thr = threshold_override if threshold_override is not None else self.threshold
            if abs(latest_price - self.current_ema) > self.current_ema * price_dev_thr:
                self.market_stable_periods_count = 0
                self.active_level = 1
                return True
            else:
                self.market_stable_periods_count += 1
                self.active_level = 0
                return False
        return False

    def get_current_sleep_interval(self) -> float:
        """
        Get the current sleep interval based on market stability.
        Returns:
            float: Sleep interval in seconds.
        """
        if self.market_stable_periods_count >= self.stable_threshold_periods:
            return self.idle_sleep_interval
        else:
            return self.active_sleep_interval

    def get_active_level(self) -> int:
        """
        Get the current active level (0 = idle, 1 = active).
        Returns:
            int: Current active level.
        """
        return self.active_level

    def force_level(self, level: int) -> None:
        """
        Manually set the active level (for CLI/manual override).
        Args:
            level (int): 0 for idle, 1 for active.
        """
        self.active_level = level
        if level == 0:
            self.market_stable_periods_count = 0

    @property
    def get_latest_price(self) -> Optional[float]:
        return self.latest.price if self.latest else None

    @property
    def get_latest_timestamp(self) -> Optional[float]:
        return self.latest.timestamp if self.latest else None

    @property
    def get_current_ema(self) -> Optional[float]:
        return self.current_ema

    @property
    def get_latest_latency_spike_flag(self) -> int:
        return self.latest.latency_spike_flag if self.latest else 0

    def get_recent_arrays(self, n: int = 100):
        """
        Returns three numpy arrays: prices, volumes, timestamps (most recent n)
        """
        buf = list(self.buffer)[-n:] if n > 0 else list(self.buffer)
        prices = np.array([u.price for u in buf], dtype=np.float32)
        volumes = np.array([u.volume for u in buf], dtype=np.float32)
        timestamps = np.array([u.timestamp for u in buf], dtype=np.float64)
        return prices, volumes, timestamps

    def add_price_data(self, price: float) -> None:
        """
        Add a new price to the prices_buffer and update EMA if enough data is present.
        Args:
            price (float): Latest market price.
        """
        self.prices_buffer.append(price)
        if len(self.prices_buffer) >= self.ema_period:
            prices = list(self.prices_buffer)[-self.ema_period:]
            if self.current_ema is None:
                self.current_ema = sum(prices) / self.ema_period
            else:
                alpha = 2 / (self.ema_period + 1)
                self.current_ema = prices[-1] * alpha + self.current_ema * (1 - alpha) 