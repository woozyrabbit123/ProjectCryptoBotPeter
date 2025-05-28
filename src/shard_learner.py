import polars as pl # Keep polars for DataFrame typing
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
import logging
import struct
import os
from dataclasses import dataclass, field # Keep field if it's intended for future use, though not currently used
from typing import Optional, Dict, Any, List, Union, Tuple # Added List, Union, Tuple
from utils.perf import timed

logger = logging.getLogger(__name__)

SHARD_FORMAT: str = '<QffBBfhBBBBfB'  # 35 bytes, must match your Wisdom Shard
SHARD_SIZE: int = struct.calcsize(SHARD_FORMAT)

SHARD_COLUMNS: List[str] = [
    'timestamp', 'price', 'volatility', 'regime_prediction', 'action_taken',
    'pnl_realized', 'micro_momentum_scaled', 'volume_divergence_flag',
    'latency_spike_flag', 'obv_trend_bullish', 'vwma_deviation_pct', 'vp_divergence_detected'
]

FEATURE_NAMES: List[str] = [
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

    def __init__(self, is_anomalous: bool, 
                 anomaly_type: Optional[str] = None, 
                 anomaly_strength: Optional[float] = None, 
                 contributing_features: Optional[Dict[str, float]] = None, 
                 confidence_penalty_suggestion: Optional[float] = None) -> None:
        self.is_anomalous = is_anomalous
        self.anomaly_type = anomaly_type
        self.anomaly_strength = anomaly_strength
        self.contributing_features = contributing_features if contributing_features is not None else {}
        self.confidence_penalty_suggestion = confidence_penalty_suggestion

class ShardLearner:
    shard_file_path: str
    config: Dict[str, Any]
    last_checked_shard_count: int

    def __init__(self, shard_file_path: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.shard_file_path: str = shard_file_path
        self.config: Dict[str, Any] = config or self.default_config()
        self.last_checked_shard_count: int = 0

    @staticmethod
    def default_config() -> Dict[str, Union[int, float]]: # More specific than Dict[str, Any]
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
        filesize: int = os.path.getsize(self.shard_file_path)
        n_shards: int = filesize // SHARD_SIZE
        if n_shards == 0:
            logger.warning("No shards to load.")
            return None
        with open(self.shard_file_path, 'rb') as f:
            data: bytes = f.read(n_shards * SHARD_SIZE)
        records: List[Tuple[Any, ...]] = [struct.unpack(SHARD_FORMAT, data[i*SHARD_SIZE:(i+1)*SHARD_SIZE]) for i in range(n_shards)]
        df: pl.DataFrame = pl.DataFrame(records, schema=SHARD_COLUMNS)
        return df

    def extract_features(self, df: pl.DataFrame) -> np.ndarray:
        prices: np.ndarray = df['price'].to_numpy()
        log_returns: np.ndarray = np.zeros_like(prices)
        if len(prices) > 1:
            log_returns[1:] = np.diff(np.log(prices))
        features: np.ndarray = np.column_stack([
            log_returns,
            df['volatility'].to_numpy(),
            df['vwma_deviation_pct'].to_numpy(),
            df['obv_trend_bullish'].to_numpy().astype(float), # Ensure boolean features are float for scaling
            df['vp_divergence_detected'].to_numpy().astype(float),
            df['latency_spike_flag'].to_numpy().astype(float)
        ])
        return features

    @timed
    def detect_anomalies(self) -> AnomalyReport:
        """
        Detect anomalies in the most recent Wisdom Shards using LOF and feature scaling.
        Failure Modes: Returns is_anomalous=False if not enough data or errors occur. May log warnings.
        Performance Notes: Uses robust scaling and LOF on a sliding window. Can be slow for very large buffers.
        """
        df: Optional[pl.DataFrame] = self.load_shards()
        min_samples: int = self.config.get('lof_min_samples', 100)
        norm_buffer_size: int = self.config.get('lof_normalization_buffer_size', 2000)
        
        if df is None or df.height < min_samples:
            logger.info("Not enough shards for anomaly detection.")
            return AnomalyReport(is_anomalous=False)
            
        df_norm: pl.DataFrame = df.tail(norm_buffer_size)
        features_norm: np.ndarray = self.extract_features(df_norm)
        
        scaler: RobustScaler = RobustScaler()
        features_scaled: np.ndarray = scaler.fit_transform(features_norm)
        
        window_size: int = self.config.get('lof_analysis_window_size', 100)
        features_window: np.ndarray = features_scaled[-window_size:]
        
        if len(features_window) < self.config['lof_n_neighbors']: # LOF n_neighbors must be < n_samples
            logger.warning(f"Not enough samples in window ({len(features_window)}) for LOF with n_neighbors={self.config['lof_n_neighbors']}.")
            return AnomalyReport(is_anomalous=False)

        lof: LocalOutlierFactor = LocalOutlierFactor(
            n_neighbors=min(self.config['lof_n_neighbors'], len(features_window) -1), # Ensure n_neighbors < n_samples_in_window
            contamination=self.config['lof_contamination'],
            novelty=False 
        )
        y_pred: np.ndarray = lof.fit_predict(features_window)
        lof_scores: np.ndarray = lof.negative_outlier_factor_
        
        is_anomalous_flag: bool = y_pred[-1] == -1
        anomaly_strength_val: Optional[float] = -lof_scores[-1] if is_anomalous_flag else None # Higher means more anomalous
        
        contributing_features_dict: Dict[str, float] = {}
        anomaly_type_str: Optional[str] = None
        confidence_penalty_val: Optional[float] = None
        
        if is_anomalous_flag:
            latest_scaled: np.ndarray = features_window[-1]
            abs_devs: np.ndarray = np.abs(latest_scaled)
            for i, val in enumerate(abs_devs):
                if val > 2.0: # Consider features with abs(z) > 2.0 as contributing
                    contributing_features_dict[FEATURE_NAMES[i]] = float(latest_scaled[i])
            
            if 'latency_spike_flag' in contributing_features_dict:
                anomaly_type_str = 'LatencyAnomaly'
                confidence_penalty_val = self.config.get('anomaly_penalty_default_latency', 0.10)
            elif 'vwma_deviation_pct' in contributing_features_dict:
                anomaly_type_str = 'VwmaDeviationAnomaly'
                confidence_penalty_val = self.config.get('anomaly_penalty_default_vwma_deviation', 0.15)
            elif 'volatility' in contributing_features_dict:
                anomaly_type_str = 'VolatilityAnomaly'
                confidence_penalty_val = self.config.get('anomaly_penalty_default_volatility', 0.15)
            elif len(contributing_features_dict) > 1:
                anomaly_type_str = 'MultiFeatureAnomaly'
                confidence_penalty_val = self.config.get('anomaly_penalty_default_multi_feature', 0.25)
            else: # Single contributing feature that is not latency, vwma, or vol
                anomaly_type_str = 'OtherAnomaly' 
                confidence_penalty_val = self.config.get('anomaly_penalty_default_multi_feature', 0.20) # Default penalty for "Other"

            logger.warning(f"Anomaly detected by ShardLearner! Type: {anomaly_type_str}, Strength: {anomaly_strength_val:.3f}, Features: {contributing_features_dict}, Penalty: {confidence_penalty_val}")
        else:
            logger.debug("No anomaly detected in the most recent shard.")
            
        return AnomalyReport(
            is_anomalous=is_anomalous_flag,
            anomaly_type=anomaly_type_str,
            anomaly_strength=anomaly_strength_val, # Explicitly None if not anomalous
            contributing_features=contributing_features_dict,
            confidence_penalty_suggestion=confidence_penalty_val
        ) 