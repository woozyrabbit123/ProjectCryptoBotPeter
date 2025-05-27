import polars as pl
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
import logging
import struct
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from utils.perf import timed

logger = logging.getLogger(__name__)

SHARD_FORMAT = '<QffBBfhBBBBfB'  # 35 bytes, must match your Wisdom Shard
SHARD_SIZE = struct.calcsize(SHARD_FORMAT)

SHARD_COLUMNS = [
    'timestamp', 'price', 'volatility', 'regime_prediction', 'action_taken',
    'pnl_realized', 'micro_momentum_scaled', 'volume_divergence_flag',
    'latency_spike_flag', 'obv_trend_bullish', 'vwma_deviation_pct', 'vp_divergence_detected'
]

FEATURE_NAMES = [
    'log_return', 'volatility', 'vwma_deviation_pct',
    'obv_trend_bullish', 'vp_divergence_detected', 'latency_spike_flag'
]

@dataclass
class AnomalyReport:
    __slots__ = [
        'is_anomalous',
        'anomaly_type',
        'anomaly_strength',
        'contributing_features',
        'confidence_penalty_suggestion'
    ]
    is_anomalous: bool
    anomaly_type: Optional[str]
    anomaly_strength: Optional[float]
    contributing_features: Dict[str, float]
    confidence_penalty_suggestion: Optional[float]

    def __init__(self, is_anomalous: bool, anomaly_type: Optional[str] = None, anomaly_strength: Optional[float] = None, contributing_features: Optional[Dict[str, float]] = None, confidence_penalty_suggestion: Optional[float] = None):
        self.is_anomalous = is_anomalous
        self.anomaly_type = anomaly_type
        self.anomaly_strength = anomaly_strength
        self.contributing_features = contributing_features if contributing_features is not None else {}
        self.confidence_penalty_suggestion = confidence_penalty_suggestion

class ShardLearner:
    def __init__(self, shard_file_path: str, config: dict = None):
        self.shard_file_path = shard_file_path
        self.config = config or self.default_config()
        self.last_checked_shard_count = 0

    @staticmethod
    def default_config() -> dict:
        return {
            'lof_analysis_window_size': 100,
            'lof_n_neighbors': 20,
            'lof_contamination': 0.05,
            'anomaly_penalty_default_latency': 0.10,
            'anomaly_penalty_default_vwma_deviation': 0.15,
            'anomaly_penalty_default_volatility': 0.15,
            'anomaly_penalty_default_multi_feature': 0.25,
            'lof_min_samples': 100,
            'lof_normalization_buffer_size': 2000,
        }

    def load_shards(self) -> Optional[pl.DataFrame]:
        if not os.path.exists(self.shard_file_path):
            logger.warning(f"Shard file {self.shard_file_path} does not exist.")
            return None
        filesize = os.path.getsize(self.shard_file_path)
        n_shards = filesize // SHARD_SIZE
        if n_shards == 0:
            logger.warning("No shards to load.")
            return None
        with open(self.shard_file_path, 'rb') as f:
            data = f.read(n_shards * SHARD_SIZE)
        records = [struct.unpack(SHARD_FORMAT, data[i*SHARD_SIZE:(i+1)*SHARD_SIZE]) for i in range(n_shards)]
        df = pl.DataFrame(records, schema=SHARD_COLUMNS)
        return df

    def extract_features(self, df: pl.DataFrame) -> np.ndarray:
        prices = df['price'].to_numpy()
        log_returns = np.zeros_like(prices)
        if len(prices) > 1:
            log_returns[1:] = np.diff(np.log(prices))
        features = np.column_stack([
            log_returns,
            df['volatility'].to_numpy(),
            df['vwma_deviation_pct'].to_numpy(),
            df['obv_trend_bullish'].to_numpy(),
            df['vp_divergence_detected'].to_numpy(),
            df['latency_spike_flag'].to_numpy()
        ])
        return features

    @timed
    def detect_anomalies(self) -> AnomalyReport:
        """
        Detect anomalies in the most recent Wisdom Shards using LOF and feature scaling.
        Failure Modes: Returns is_anomalous=False if not enough data or errors occur. May log warnings.
        Performance Notes: Uses robust scaling and LOF on a sliding window. Can be slow for very large buffers.
        """
        df = self.load_shards()
        min_samples = self.config.get('lof_min_samples', 100)
        norm_buffer_size = self.config.get('lof_normalization_buffer_size', 2000)
        if df is None or df.height < min_samples:
            logger.info("Not enough shards for anomaly detection.")
            return AnomalyReport(is_anomalous=False)
        # Use a larger buffer for normalization, but only the last N for LOF
        df_norm = df.tail(norm_buffer_size)
        features_norm = self.extract_features(df_norm)
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_norm)
        # Only use the last lof_analysis_window_size for LOF
        window_size = self.config.get('lof_analysis_window_size', 100)
        features_window = features_scaled[-window_size:]
        lof = LocalOutlierFactor(
            n_neighbors=self.config['lof_n_neighbors'],
            contamination=self.config['lof_contamination'],
            novelty=False
        )
        y_pred = lof.fit_predict(features_window)
        lof_scores = lof.negative_outlier_factor_
        # Check if the most recent shard is anomalous
        is_anomalous = y_pred[-1] == -1
        anomaly_strength = -lof_scores[-1]  # Higher means more anomalous
        contributing_features = {}
        anomaly_type = None
        confidence_penalty_suggestion = None
        if is_anomalous:
            # Find which features are most extreme for the latest point
            latest_scaled = features_window[-1]
            abs_devs = np.abs(latest_scaled)
            # Consider features with abs(z) > 2 as contributing
            for i, val in enumerate(abs_devs):
                if val > 2.0:
                    contributing_features[FEATURE_NAMES[i]] = float(latest_scaled[i])
            # Classify anomaly type
            if 'latency_spike_flag' in contributing_features:
                anomaly_type = 'LatencyAnomaly'
                confidence_penalty_suggestion = self.config.get('anomaly_penalty_default_latency', 0.10)
            elif 'vwma_deviation_pct' in contributing_features:
                anomaly_type = 'VwmaDeviationAnomaly'
                confidence_penalty_suggestion = self.config.get('anomaly_penalty_default_vwma_deviation', 0.15)
            elif 'volatility' in contributing_features:
                anomaly_type = 'VolatilityAnomaly'
                confidence_penalty_suggestion = self.config.get('anomaly_penalty_default_volatility', 0.15)
            elif len(contributing_features) > 1:
                anomaly_type = 'MultiFeatureAnomaly'
                confidence_penalty_suggestion = self.config.get('anomaly_penalty_default_multi_feature', 0.25)
            else:
                anomaly_type = 'OtherAnomaly'
                confidence_penalty_suggestion = self.config.get('anomaly_penalty_default_multi_feature', 0.25)
            logger.warning(f"Anomaly detected by ShardLearner! Type: {anomaly_type}, Strength: {anomaly_strength:.3f}, Features: {contributing_features}, Penalty: {confidence_penalty_suggestion}")
        else:
            logger.debug("No anomaly detected in the most recent shard.")
        return AnomalyReport(
            is_anomalous=is_anomalous,
            anomaly_type=anomaly_type,
            anomaly_strength=anomaly_strength if is_anomalous else None,
            contributing_features=contributing_features,
            confidence_penalty_suggestion=confidence_penalty_suggestion
        ) 