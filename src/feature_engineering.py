"""
Feature engineering module for Project Crypto Bot Peter.
Handles calculation of technical indicators and feature preparation for the model.
"""

import polars as pl
import numpy as np
from typing import Optional
import logging
from dataclasses import dataclass
import collections
import time
from utils.perf import timed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: Implement feature calculation functions 

class EMA:
    def __init__(self, span: int, smoothing: float = 2.0):
        self.span = span
        self.alpha = smoothing / (1.0 + span)
        self.current_ema: Optional[float] = None
        self.num_observations = 0

    def update(self, price: float) -> Optional[float]:
        if self.current_ema is None:
            self.current_ema = price
        else:
            self.current_ema = (price * self.alpha) + (self.current_ema * (1.0 - self.alpha))
        self.num_observations += 1
        return self.current_ema

    @property
    def value(self) -> Optional[float]:
        return self.current_ema

class VolatilityRegime:
    def __init__(self, window: int = 34, price_buffer_ref=None):
        self.window = window
        self.price_buffer_ref = price_buffer_ref
        self.thresholds = {'low_raw': 0.0001, 'medium_raw': 0.0005}
        self.last_raw_volatility = None

    def classify_current(self) -> str:
        buf = self.price_buffer_ref
        if buf is None or len(buf) < max(10, self.window // 2):
            self.last_raw_volatility = None
            return "unknown"
        if isinstance(buf, collections.deque):
            prices = list(buf)[-self.window:]
        else:
            prices = buf.get_recent(self.window)[0]
        prices = np.array(prices, dtype=np.float64)
        if len(prices) < max(10, self.window // 2):
            self.last_raw_volatility = None
            return "unknown"
        log_returns = np.diff(np.log(prices))
        current_vol = np.std(log_returns)
        self.last_raw_volatility = float(current_vol)
        if current_vol < self.thresholds['low_raw']:
            return "low"
        elif current_vol < self.thresholds['medium_raw']:
            return "medium"
        else:
            return "high"

@dataclass
class RegimeSignal:
    __slots__ = [
        'trend_strength',
        'volatility_regime',
        'confidence',
        'timestamp',
        'obv_value',
        'obv_trend_bullish',
        'vwma_deviation_pct',
        'vp_divergence_detected'
    ]
    trend_strength: float
    volatility_regime: str
    confidence: float
    timestamp: float
    obv_value: float
    obv_trend_bullish: bool
    vwma_deviation_pct: float
    vp_divergence_detected: bool

    def __init__(self, trend_strength: float, volatility_regime: str, confidence: float, timestamp: float, obv_value: float = 0.0, obv_trend_bullish: bool = False, vwma_deviation_pct: float = 0.0, vp_divergence_detected: bool = False):
        self.trend_strength = trend_strength
        self.volatility_regime = volatility_regime
        self.confidence = confidence
        self.timestamp = timestamp
        self.obv_value = obv_value
        self.obv_trend_bullish = obv_trend_bullish
        self.vwma_deviation_pct = vwma_deviation_pct
        self.vp_divergence_detected = vp_divergence_detected

class MicroRegimeDetector:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or self.default_config()
        self.fast_ema = EMA(self.config['fast_span'])
        self.med_ema = EMA(self.config['med_span'])
        self.slow_ema = EMA(self.config['slow_span'])
        self.vol_regime = VolatilityRegime(window=self.config['vol_window'])
        self.samples_count = 0
        self._prices = None
        self._volumes = None
        self._timestamps = None
        # OBV state
        self._last_obv = 0.0
        self._obv_short_ema = None
        self._obv_long_ema = None

    @staticmethod
    def default_config():
        return {
            'fast_span': 8,
            'med_span': 21,
            'slow_span': 55,
            'vol_window': 34,
            'obv_short_ema': 7,
            'obv_long_ema': 21,
            'obv_div_lookback': 30,
            'vwma_window': 20,
            'vp_divergence_price_stagnation_threshold_pct': 0.1,  # percent
            'vp_divergence_volume_spike_multiplier': 1.5,
            'vp_divergence_lookback': 5,
        }

    def update_buffer(self, prices: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray):
        """
        Update the internal buffer arrays for price, volume, and timestamp.
        """
        self._prices = prices
        self._volumes = volumes
        self._timestamps = timestamps
        self.vol_regime.price_buffer_ref = prices  # For volatility regime

    def _calc_obv(self):
        if self._prices is None or self._volumes is None or len(self._prices) < 2:
            return 0.0, []
        obv = [0.0]
        for i in range(1, len(self._prices)):
            if self._prices[i] > self._prices[i-1]:
                obv.append(obv[-1] + self._volumes[i])
            elif self._prices[i] < self._prices[i-1]:
                obv.append(obv[-1] - self._volumes[i])
            else:
                obv.append(obv[-1])
        return obv[-1], obv

    def _ema(self, arr, span):
        if len(arr) == 0:
            return 0.0
        alpha = 2 / (span + 1)
        ema = arr[0]
        for v in arr[1:]:
            ema = v * alpha + ema * (1 - alpha)
        return ema

    def _calc_obv_trend(self, obv_arr):
        short_ema = self._ema(obv_arr[-self.config['obv_short_ema']:], self.config['obv_short_ema'])
        long_ema = self._ema(obv_arr[-self.config['obv_long_ema']:], self.config['obv_long_ema'])
        return short_ema > long_ema

    def _calc_vwma(self):
        window = self.config['vwma_window']
        if self._prices is None or self._volumes is None or len(self._prices) < window:
            return 0.0, 0.0
        p = self._prices[-window:]
        v = self._volumes[-window:]
        vwma = np.sum(p * v) / np.sum(v) if np.sum(v) > 0 else 0.0
        deviation = ((self._prices[-1] - vwma) / vwma) * 100 if vwma != 0 else 0.0
        return vwma, deviation

    def _calc_vp_divergence(self):
        lookback = self.config['vp_divergence_lookback']
        price_thresh = self.config['vp_divergence_price_stagnation_threshold_pct']
        vol_mult = self.config['vp_divergence_volume_spike_multiplier']
        if self._prices is None or self._volumes is None or len(self._prices) < lookback + 1:
            return False
        price_change = abs((self._prices[-1] - self._prices[-lookback]) / self._prices[-lookback]) * 100
        avg_vol = np.mean(self._volumes[-lookback-1:-1])
        curr_vol = self._volumes[-1]
        if price_change < price_thresh and curr_vol > vol_mult * avg_vol:
            return True
        return False

    @timed
    def detect(self) -> 'RegimeSignal':
        """
        Run detection using the most recent price in the buffer.
        Failure Modes: Returns RegimeSignal with confidence=0.0 if not enough data.
        Performance Notes: Fast for typical buffer sizes; may slow with very large arrays.
        """
        if self._prices is None or len(self._prices) == 0:
            # Not enough data
            return RegimeSignal(
                trend_strength=0.0,
                volatility_regime="unknown",
                confidence=0.0,
                timestamp=time.time(),
                obv_value=0.0,
                obv_trend_bullish=False,
                vwma_deviation_pct=0.0,
                vp_divergence_detected=False
            )
        current_price = self._prices[-1]
        self.fast_ema.update(current_price)
        self.med_ema.update(current_price)
        self.slow_ema.update(current_price)
        self.samples_count += 1
        trend_alignment = self.calculate_alignment(self.fast_ema.value, self.med_ema.value, self.slow_ema.value)
        vol_regime_str = self.vol_regime.classify_current()
        ema_sep = 0.0
        if self.fast_ema.value and self.slow_ema.value:
            ema_sep = abs(self.fast_ema.value - self.slow_ema.value) / (abs(self.slow_ema.value) if self.slow_ema.value else 1.0)
        confidence = self.calculate_confidence(ema_sep)
        # --- Volume features ---
        obv_value, obv_arr = self._calc_obv()
        obv_trend_bullish = self._calc_obv_trend(obv_arr) if len(obv_arr) >= self.config['obv_long_ema'] else False
        _, vwma_deviation = self._calc_vwma()
        vp_divergence = self._calc_vp_divergence()
        logger.debug(f"OBV: {obv_value}, OBV trend bullish: {obv_trend_bullish}, VWMA deviation: {vwma_deviation:.2f}%, VP divergence: {vp_divergence}")
        return RegimeSignal(
            trend_strength=trend_alignment,
            volatility_regime=vol_regime_str,
            confidence=confidence,
            timestamp=time.time(),
            obv_value=obv_value,
            obv_trend_bullish=obv_trend_bullish,
            vwma_deviation_pct=vwma_deviation,
            vp_divergence_detected=vp_divergence
        )

    def calculate_alignment(self, fast_val, med_val, slow_val) -> float:
        # Use np.tanh and normalization for nuanced trend strength
        if fast_val is None or med_val is None or slow_val is None:
            return 0.0
        vals = np.array([fast_val, med_val, slow_val], dtype=np.float64)
        diffs = np.diff(vals)
        # Normalize by slow_val to avoid scale issues
        norm = abs(slow_val) if slow_val else 1.0
        alignment = np.tanh(np.sum(diffs) / norm)
        # Strong uptrend: fast > med > slow, strong downtrend: fast < med < slow
        if fast_val > med_val > slow_val:
            return alignment
        elif fast_val < med_val < slow_val:
            return -alignment
        else:
            return 0.0

    def calculate_confidence(self, ema_separation_factor: float) -> float:
        # Composite: EMA separation, placeholder vol_consistency, data freshness
        vol_consistency = 0.5  # Placeholder for now
        data_freshness = min(1.0, self.samples_count / 100)
        # Weighted sum (tune weights as needed)
        confidence = 0.5 * ema_separation_factor + 0.2 * vol_consistency + 0.3 * data_freshness
        return min(1.0, max(0.0, confidence)) 