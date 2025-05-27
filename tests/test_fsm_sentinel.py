import pytest
from src.fsm_sentinel import SleepySentinelFSM


def test_initialization_defaults():
    fsm = SleepySentinelFSM()
    assert fsm.threshold == 0.0005
    assert fsm.ema_period == 5
    assert fsm.stable_threshold_periods == 100
    assert fsm.active_sleep_interval == 0.1
    assert fsm.idle_sleep_interval == 0.5
    assert fsm.market_stable_periods_count == 0
    assert fsm.current_ema is None
    assert fsm.active_level == 0
    assert isinstance(fsm.prices_buffer, type(fsm.prices_buffer))


def test_initialization_custom():
    fsm = SleepySentinelFSM(threshold=0.01, ema_period=10, stable_threshold_periods=50, active_sleep=0.2, idle_sleep=1.0)
    assert fsm.threshold == 0.01
    assert fsm.ema_period == 10
    assert fsm.stable_threshold_periods == 50
    assert fsm.active_sleep_interval == 0.2
    assert fsm.idle_sleep_interval == 1.0


def test_add_price_data_and_ema():
    fsm = SleepySentinelFSM(ema_period=3)
    prices = [100, 101, 102]
    for p in prices:
        fsm.add_price_data(p)
    assert len(fsm.prices_buffer) == 3
    assert fsm.current_ema is not None
    # Add another price and check EMA updates
    prev_ema = fsm.current_ema
    fsm.add_price_data(103)
    assert fsm.current_ema != prev_ema


def test_check_for_wake_trigger_stable():
    fsm = SleepySentinelFSM(ema_period=3, threshold=0.05)
    # Add stable prices
    for p in [100, 100, 100]:
        fsm.add_price_data(p)
    assert fsm.check_for_wake_trigger() is False
    assert fsm.active_level == 0
    # Add more stable prices
    for _ in range(5):
        fsm.add_price_data(100)
        fsm.check_for_wake_trigger()
    assert fsm.market_stable_periods_count > 0


def test_check_for_wake_trigger_trigger():
    fsm = SleepySentinelFSM(ema_period=3, threshold=0.01)
    # Add stable prices
    for p in [100, 100, 100]:
        fsm.add_price_data(p)
    # Add a price that exceeds the threshold
    fsm.add_price_data(110)
    triggered = fsm.check_for_wake_trigger()
    assert triggered is True
    assert fsm.active_level == 1
    assert fsm.market_stable_periods_count == 0


def test_market_stable_periods_count_resets_on_trigger():
    fsm = SleepySentinelFSM(ema_period=3, threshold=0.01)
    for p in [100, 100, 100]:
        fsm.add_price_data(p)
    # Stable period
    fsm.check_for_wake_trigger()
    fsm.check_for_wake_trigger()
    assert fsm.market_stable_periods_count > 0
    # Trigger
    fsm.add_price_data(110)
    fsm.check_for_wake_trigger()
    assert fsm.market_stable_periods_count == 0


def test_get_current_sleep_interval():
    fsm = SleepySentinelFSM(stable_threshold_periods=3)
    # Not stable yet
    fsm.market_stable_periods_count = 2
    assert fsm.get_current_sleep_interval() == fsm.active_sleep_interval
    # Now stable
    fsm.market_stable_periods_count = 3
    assert fsm.get_current_sleep_interval() == fsm.idle_sleep_interval


def test_force_level():
    fsm = SleepySentinelFSM()
    fsm.force_level(1)
    assert fsm.active_level == 1
    fsm.market_stable_periods_count = 5
    fsm.force_level(0)
    assert fsm.active_level == 0
    assert fsm.market_stable_periods_count == 0 