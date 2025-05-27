import numpy as np

class CES_v1_0:
    """
    Contextual Environmental Score (CES) v1.0
    Calculates a vector of normalized market environment features for use in evolutionary and meta-learning systems.
    """
    def __init__(self, volatility_lookback=20, trend_lookback=10, liquidity_lookback=10, weights=None):
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
            'mle_confidence': 0.0
        }

    def calculate_ces_vector(self, current_market_data, optional_mle_input=None):
        """
        Calculate the CES vector from current_market_data (expects dict with 'ohlcv' DataFrame).
        Returns a dict of normalized scores for each dimension.
        """
        ohlcv = current_market_data.get('ohlcv')
        ces_vector = {}

        # Volatility: Realized volatility (std of log returns)
        if ohlcv is not None and len(ohlcv) > 1:
            close = ohlcv['close'].values[-self.volatility_lookback:]
            log_returns = np.diff(np.log(close))
            realized_vol = np.std(log_returns) * np.sqrt(365*24)  # annualized hourly
            # Normalize: sigmoid for now
            v_norm = 1 / (1 + np.exp(-10 * (realized_vol - 0.01)))
        else:
            v_norm = 0.5  # neutral
        ces_vector['volatility'] = v_norm

        # Liquidity: Use volume as a simple proxy (normalized by rolling max)
        if ohlcv is not None and len(ohlcv) >= self.liquidity_lookback:
            volume = ohlcv['volume'].values[-self.liquidity_lookback:]
            l_raw = np.mean(volume)
            # Normalize: divide by max in window (avoid div0)
            l_norm = l_raw / (np.max(volume) + 1e-8)
        else:
            l_norm = 0.5
        ces_vector['liquidity'] = l_norm

        # Trend/Momentum: Slope of SMA over trend_lookback
        if ohlcv is not None and len(ohlcv) >= self.trend_lookback:
            closes = ohlcv['close'].values[-self.trend_lookback:]
            x = np.arange(len(closes))
            # Linear fit slope
            slope = np.polyfit(x, closes, 1)[0]
            # Normalize: sigmoid
            t_norm = 1 / (1 + np.exp(-slope/0.1))
        else:
            t_norm = 0.5
        ces_vector['trend'] = t_norm

        # Inter-market correlations: Placeholder (neutral)
        ces_vector['inter_market_corr'] = 0.5

        # Anomaly signals: Placeholder (neutral)
        ces_vector['anomaly'] = 0.5

        # MLE input: pass through if present
        if optional_mle_input and 'pattern_regime_confidence' in optional_mle_input:
            ces_vector['mle_confidence'] = optional_mle_input['pattern_regime_confidence']
        else:
            ces_vector['mle_confidence'] = 0.5

        return ces_vector

    def get_composite_stress_score(self, ces_vector):
        """
        Calculate a single composite score from the CES vector using static weights.
        """
        score = 0.0
        for k, w in self.weights.items():
            score += w * ces_vector.get(k, 0.5)
        return score 