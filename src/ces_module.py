"""
Contextual Environment Scorer (CES) for Project Crypto Bot Peter.

This module defines the `CES_v1_0` class, which is responsible for
assessing the current market environment. It calculates a "Contextual
Environmental Score" (CES) vector based on various market features
derived from OHLCV data, such as volatility, trend, and liquidity.
This vector provides a quantitative representation of the market's
state, which can be used by other components of the trading system
(like the Logic Evolution Engine or System Orchestrator) to adapt
their behavior or decision-making processes.
"""
import numpy as np
from typing import Dict, Optional, Any, List # Added typing imports
import pandas as pd # Assuming ohlcv is a DataFrame

class CES_v1_0:
    """
    Contextual Environmental Score (CES) v1.0
    Calculates a vector of normalized market environment features for use in evolutionary and meta-learning systems.
    """
    volatility_lookback: int
    trend_lookback: int
    liquidity_lookback: int
    weights: Dict[str, float]

    def __init__(self, volatility_lookback: int = 20, trend_lookback: int = 10, liquidity_lookback: int = 10, weights: Optional[Dict[str, float]] = None) -> None:
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.liquidity_lookback = liquidity_lookback
        # Static weights for composite score (can be tuned)
        self.weights = weights or {
            'volatility': 0.3,
            'liquidity': 0.2,
            'trend': 0.3,
            'inter_market_corr': 0.1,
            'anomaly': 0.1,
            'mle_confidence': 0.0 # Ensure this matches potential keys in ces_vector
        }

    def calculate_ces_vector(self, current_market_data: Dict[str, Any], optional_mle_input: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate the CES vector from current_market_data (expects dict with 'ohlcv' DataFrame).
        Returns a dict of normalized scores for each dimension.
        """
        ohlcv: Optional[pd.DataFrame] = current_market_data.get('ohlcv')
        ces_vector: Dict[str, float] = {}

        v_norm: float
        # Volatility: Realized volatility (std of log returns)
        if ohlcv is not None and not ohlcv.empty and 'close' in ohlcv.columns and len(ohlcv) > 1: # Added more checks
            close_prices: np.ndarray = ohlcv['close'].values[-self.volatility_lookback:]
            if len(close_prices) > 1: # Ensure enough data points for diff
                log_returns: np.ndarray = np.diff(np.log(close_prices))
                realized_vol: float = np.std(log_returns) * np.sqrt(365*24)  # annualized hourly
                v_norm = 1 / (1 + np.exp(-10 * (realized_vol - 0.01))) # Normalize: sigmoid for now
            else:
                v_norm = 0.5 # Not enough data
        else:
            v_norm = 0.5  # neutral if no data or not enough data
        ces_vector['volatility'] = v_norm

        l_norm: float
        # Liquidity: Use volume as a simple proxy (normalized by rolling max)
        if ohlcv is not None and not ohlcv.empty and 'volume' in ohlcv.columns and len(ohlcv) >= self.liquidity_lookback:
            volume_data: np.ndarray = ohlcv['volume'].values[-self.liquidity_lookback:]
            if len(volume_data) > 0:
                l_raw: float = np.mean(volume_data)
                max_volume: float = np.max(volume_data)
                l_norm = l_raw / (max_volume + 1e-8) if max_volume > 0 else 0.5 # Normalize: divide by max in window (avoid div0)
            else:
                l_norm = 0.5 # Not enough data
        else:
            l_norm = 0.5 # neutral
        ces_vector['liquidity'] = l_norm
        
        t_norm: float
        # Trend/Momentum: Slope of SMA over trend_lookback
        if ohlcv is not None and not ohlcv.empty and 'close' in ohlcv.columns and len(ohlcv) >= self.trend_lookback:
            closes: np.ndarray = ohlcv['close'].values[-self.trend_lookback:]
            if len(closes) >= 2: # np.polyfit needs at least 2 points for degree 1
                x: np.ndarray = np.arange(len(closes))
                slope: float = np.polyfit(x, closes, 1)[0]
                t_norm = 1 / (1 + np.exp(-slope/0.1)) # Normalize: sigmoid
            else:
                t_norm = 0.5 # Not enough data
        else:
            t_norm = 0.5 # neutral
        ces_vector['trend'] = t_norm

        # Inter-market correlations: Placeholder (neutral)
        ces_vector['inter_market_corr'] = 0.5

        # Anomaly signals: Placeholder (neutral)
        ces_vector['anomaly'] = 0.5

        # MLE input: pass through if present
        mle_conf: float = 0.5 # Default
        if optional_mle_input and 'pattern_regime_confidence' in optional_mle_input:
            mle_conf_val = optional_mle_input['pattern_regime_confidence']
            if isinstance(mle_conf_val, (float, int)): # Basic type check
                 mle_conf = float(mle_conf_val)
        ces_vector['mle_confidence'] = mle_conf


        return ces_vector

    def get_composite_stress_score(self, ces_vector: Dict[str, float]) -> float:
        """
        Calculate a single composite score from the CES vector using static weights.
        """
        score: float = 0.0
        for k, w in self.weights.items():
            score += w * ces_vector.get(k, 0.5) # Default to 0.5 if key is missing
        return score 