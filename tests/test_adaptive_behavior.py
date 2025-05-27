import pytest
import numpy as np
from src.adaptive_behavior import (
    calculate_persona, calculate_env_score, AnomalySequenceDetector,
    OpportunityActionExecutor, MarketPersona, detect_hunter_momentum_burst
)

class DummyConfig:
    def getfloat(self, section, key, fallback=None):
        return fallback if fallback is not None else 1.0
    def getint(self, section, key, fallback=None):
        return fallback if fallback is not None else 1
    def get(self, section, key, fallback=None):
        return fallback if fallback is not None else ''
    def __getitem__(self, key):
        return self

@pytest.fixture
def dummy_config():
    return DummyConfig()

def test_calculate_persona_basic(dummy_config):
    # Should return GUARDIAN for neutral inputs
    persona = calculate_persona([], 0.0, 0.5, 0.5, dummy_config)
    assert persona in [MarketPersona.HUNTER, MarketPersona.GUARDIAN, MarketPersona.GHOST]

def test_calculate_env_score_basic(dummy_config):
    score = calculate_env_score([], [], [], MarketPersona.GUARDIAN, dummy_config)
    assert 0.0 <= score <= 1.0

def test_anomaly_sequence_detector_update():
    config = DummyConfig()
    seq_conf = {'sequences_of_interest': 'A->B->C', 'suggested_responses': 'A->B->C:{"action":"log_only"}'}
    detector = AnomalySequenceDetector(seq_conf, config)
    assert detector.update('A') is None
    assert detector.update('B') is None
    result = detector.update('C')
    assert result is not None
    assert 'sequence' in result

def test_opportunity_action_executor_vectorized():
    config = DummyConfig()
    executor = OpportunityActionExecutor(config)
    opp_signal = {'opportunity_type': 'sharp_vwma_deviation', 'vwma_deviation': 0.05}
    base_params = {'base_confidence': 0.8, 'env_score': 0.2, 'size': 1.0, 'tp': 1.0, 'sl': 1.0}
    persona = MarketPersona.HUNTER
    config_dict = {'OpportunitySafetyLimits': {'max_opportunity_actions_per_hour': 2, 'opportunity_min_base_confidence_floor': 0.3, 'max_param_mod_amplitude_factor': 2.0}}
    result = executor.execute_advanced_opportunity(opp_signal, base_params, persona, config_dict)
    assert isinstance(result, dict)

def test_detect_hunter_momentum_burst():
    config = DummyConfig()
    # 15 prices, 3 consecutive 5-period windows with >3% rise
    prices = np.array([1,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.10,1.11,1.12,1.13,1.14], dtype=np.float32)
    assert detect_hunter_momentum_burst(prices, config) is True
    # Flat prices, should not detect
    prices = np.ones(15, dtype=np.float32)
    assert detect_hunter_momentum_burst(prices, config) is False 