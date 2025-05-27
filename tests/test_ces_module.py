import pytest
import pandas as pd
import numpy as np
from src.ces_module import CES_v1_0

def make_mock_ohlcv(n=30):
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'),
        'open': np.linspace(100, 110, n),
        'high': np.linspace(101, 111, n),
        'low': np.linspace(99, 109, n),
        'close': np.linspace(100, 110, n) + np.random.normal(0, 0.5, n),
        'volume': np.linspace(1000, 2000, n) + np.random.normal(0, 50, n)
    })

def test_ces_init():
    ces = CES_v1_0()
    assert isinstance(ces, CES_v1_0)
    assert hasattr(ces, 'volatility_lookback')
    assert hasattr(ces, 'weights')

def test_calculate_ces_vector_keys():
    ces = CES_v1_0()
    ohlcv = make_mock_ohlcv()
    market_data = {'ohlcv': ohlcv}
    ces_vec = ces.calculate_ces_vector(market_data)
    expected_keys = {'volatility', 'liquidity', 'trend', 'inter_market_corr', 'anomaly', 'mle_confidence'}
    assert set(ces_vec.keys()) == expected_keys
    for v in ces_vec.values():
        assert isinstance(v, float)

def test_calculate_ces_vector_with_mle_input():
    ces = CES_v1_0()
    ohlcv = make_mock_ohlcv()
    market_data = {'ohlcv': ohlcv}
    mle_input = {'pattern_regime_confidence': 0.77}
    ces_vec = ces.calculate_ces_vector(market_data, optional_mle_input=mle_input)
    assert ces_vec['mle_confidence'] == 0.77

def test_get_composite_stress_score():
    ces = CES_v1_0()
    ohlcv = make_mock_ohlcv()
    market_data = {'ohlcv': ohlcv}
    ces_vec = ces.calculate_ces_vector(market_data)
    score = ces.get_composite_stress_score(ces_vec)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0 