import unittest
import numpy as np
from typing import List, Dict, Any, Optional

# Attempt to import the functions to be tested
try:
    from src.utils.performance_metrics_utils import (
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_trade_stats
    )
except ImportError:
    # Fallback for environments where PYTHONPATH might not be perfectly set for test discovery
    # The tests will fail if the functions truly cannot be imported at runtime by the test runner
    calculate_sharpe_ratio = None
    calculate_max_drawdown = None
    calculate_trade_stats = None
    print("Warning: Utility functions could not be imported from src.utils.performance_metrics_utils. Tests may fail.")


class TestPerformanceMetricsUtils(unittest.TestCase):
    """
    Test suite for the performance metric utility functions.
    """

    def setUp(self) -> None:
        """
        Set up for test methods. This method will be called before every test.
        """
        if not all([calculate_sharpe_ratio, calculate_max_drawdown, calculate_trade_stats]):
            self.fail("One or more utility functions not imported. Cannot run tests.")

    # --- Tests for calculate_sharpe_ratio ---
    def test_sharpe_ratio_normal_case(self) -> None:
        equity_curve: List[float] = [100, 105, 102, 108, 110, 107, 112] # Example data
        # Expected value needs to be calculated based on the formula or a reliable source for these inputs.
        # For this example, let's assume a hypothetical expected value.
        # Returns: 0.05, -0.02857, 0.0588, 0.0185, -0.02727, 0.0467
        # Mean: 0.0197
        # Std: 0.0358
        # Sharpe (annualized): (0.0197 / 0.0358) * sqrt(252) approx 8.73
        expected_sharpe = (np.mean([0.05, -0.0285714, 0.0588235, 0.0185185, -0.0272727, 0.0467289]) / 
                           np.std([0.05, -0.0285714, 0.0588235, 0.0185185, -0.0272727, 0.0467289])) * np.sqrt(252)
        self.assertAlmostEqual(calculate_sharpe_ratio(equity_curve), expected_sharpe, places=4)

    def test_sharpe_ratio_empty_curve(self) -> None:
        self.assertTrue(np.isnan(calculate_sharpe_ratio([])))

    def test_sharpe_ratio_insufficient_data(self) -> None:
        self.assertTrue(np.isnan(calculate_sharpe_ratio([100.0])))

    def test_sharpe_ratio_no_std_dev(self) -> None:
        # Flat line (no change in returns -> std dev of returns is 0)
        self.assertTrue(np.isnan(calculate_sharpe_ratio([100, 100, 100, 100])))
        # Linearly increasing equity (constant return -> std dev of returns is 0)
        # Returns: 0.01, 0.00990099, 0.0098039... these are not constant.
        # A better test: constant returns from constant equity values.
        # If equity is [100, 101, 102.01, 103.0301] (1% constant return)
        # The function calculates returns on equity_curve, not on price directly.
        # So, if returns are [0.01, 0.01, 0.01], std_dev_returns will be 0.
        # To engineer this, we need an equity curve where (E_t / E_{t-1}) -1 is constant.
        # e.g. [100, 101, 102.01, 103.0301] gives returns [0.01, 0.01, 0.01]
        # No, np.diff(equity_array)/equity_array[:-1] for [100,100,100] is [0,0], std is 0.
        # For [100, 101, 102, 103], returns = [0.01, 0.0099, 0.0098], std is not 0.
        # The current implementation of sharpe ratio will have std_dev_returns = 0 if all returns are identical.
        # This happens if all price changes are zero, e.g., equity_curve = [100, 100, 100].
        self.assertTrue(np.isnan(calculate_sharpe_ratio([100.0, 100.0, 100.0])))


    def test_sharpe_ratio_all_losses(self) -> None:
        equity_curve: List[float] = [100, 90, 80, 70]
        # Returns: -0.1, -0.1111, -0.125
        # Mean: -0.1120
        # Std: 0.0096
        # Sharpe: (-0.1120 / 0.0096) * sqrt(252) approx -185
        expected_sharpe = (np.mean([-0.1, -0.111111, -0.125]) / 
                           np.std([-0.1, -0.111111, -0.125])) * np.sqrt(252)
        self.assertAlmostEqual(calculate_sharpe_ratio(equity_curve), expected_sharpe, places=4)
        self.assertLess(calculate_sharpe_ratio(equity_curve), 0)

    def test_sharpe_ratio_all_gains(self) -> None:
        equity_curve: List[float] = [100, 110, 121, 133.1] # Consistent 10% gain
        # Returns: 0.1, 0.1, 0.1. Std dev of returns is 0.
        # This case should return NaN as per current logic for std_dev_returns == 0
        # If the function were to handle this (e.g., by returning a large positive number), this test would change.
        self.assertTrue(np.isnan(calculate_sharpe_ratio(equity_curve)), "Sharpe should be NaN for zero std dev of returns")


    # --- Tests for calculate_max_drawdown ---
    def test_max_drawdown_normal_case(self) -> None:
        equity_curve: List[float] = [100, 120, 90, 110, 80, 130, 100]
        # Peaks: 100, 120, 120, 120, 120, 130, 130
        # Troughs: 100, 120, 90, 110, 80, 130, 100
        # Drawdowns from peak:
        # Peak 120: (120-90)/120 = 0.25 (25%)
        # Peak 120: (120-80)/120 = 0.3333 (33.33%)
        # Peak 130: (130-100)/130 = 0.2307 (23.07%)
        expected_max_dd = (120.0 - 80.0) / 120.0 * 100.0
        self.assertAlmostEqual(calculate_max_drawdown(equity_curve), expected_max_dd, places=4)

    def test_max_drawdown_empty_curve(self) -> None:
        self.assertTrue(np.isnan(calculate_max_drawdown([])))

    def test_max_drawdown_no_drawdown(self) -> None:
        equity_curve: List[float] = [100, 110, 120, 130]
        self.assertAlmostEqual(calculate_max_drawdown(equity_curve), 0.0, places=4)

    def test_max_drawdown_full_loss(self) -> None:
        equity_curve: List[float] = [100, 50, 10, 1]
        # Peak 100: (100-1)/100 = 0.99
        expected_max_dd = (100.0 - 1.0) / 100.0 * 100.0
        self.assertAlmostEqual(calculate_max_drawdown(equity_curve), expected_max_dd, places=4)
        
    def test_max_drawdown_starts_low(self) -> None:
        equity_curve: List[float] = [10, 50, 20, 60, 10]
        # Peak 10 -> no dd
        # Peak 50 -> (50-20)/50 = 0.6 (60%)
        # Peak 60 -> (60-10)/60 = 0.8333 (83.33%)
        expected_max_dd = (60.0-10.0)/60.0 * 100
        self.assertAlmostEqual(calculate_max_drawdown(equity_curve), expected_max_dd, places=4)

    # --- Tests for calculate_trade_stats ---
    def test_trade_stats_normal_case(self) -> None:
        trade_log: List[Dict[str, Any]] = [
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 110}, # P/L = +10 (Win)
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 90},  # P/L = -10 (Loss)
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 120}, # P/L = +20 (Win)
        ]
        # Wins: 2, Losses: 1. Total: 3
        # Win %: (2/3)*100 = 66.666...%
        # Avg P/L: (10 - 10 + 20) / 3 = 20 / 3 = 6.666...
        expected_stats = {
            'winning_trade_percentage': (2.0/3.0) * 100.0,
            'average_trade_pl': (10.0 - 10.0 + 20.0) / 3.0,
            'total_trades': 3.0
        }
        result = calculate_trade_stats(trade_log)
        self.assertAlmostEqual(result['winning_trade_percentage'], expected_stats['winning_trade_percentage'], places=4)
        self.assertAlmostEqual(result['average_trade_pl'], expected_stats['average_trade_pl'], places=4)
        self.assertEqual(result['total_trades'], expected_stats['total_trades'])

    def test_trade_stats_empty_log(self) -> None:
        expected_stats = {'winning_trade_percentage': np.nan, 'average_trade_pl': np.nan, 'total_trades': 0.0}
        result = calculate_trade_stats([])
        self.assertTrue(np.isnan(result['winning_trade_percentage']))
        self.assertTrue(np.isnan(result['average_trade_pl']))
        self.assertEqual(result['total_trades'], expected_stats['total_trades'])

    def test_trade_stats_all_wins(self) -> None:
        trade_log: List[Dict[str, Any]] = [
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 110},
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 120},
        ]
        expected_stats = {'winning_trade_percentage': 100.0, 'average_trade_pl': 15.0, 'total_trades': 2.0}
        result = calculate_trade_stats(trade_log)
        self.assertAlmostEqual(result['winning_trade_percentage'], expected_stats['winning_trade_percentage'], places=4)
        self.assertAlmostEqual(result['average_trade_pl'], expected_stats['average_trade_pl'], places=4)
        self.assertEqual(result['total_trades'], expected_stats['total_trades'])

    def test_trade_stats_all_losses(self) -> None:
        trade_log: List[Dict[str, Any]] = [
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 90},
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 80},
        ]
        expected_stats = {'winning_trade_percentage': 0.0, 'average_trade_pl': -15.0, 'total_trades': 2.0}
        result = calculate_trade_stats(trade_log)
        self.assertAlmostEqual(result['winning_trade_percentage'], expected_stats['winning_trade_percentage'], places=4)
        self.assertAlmostEqual(result['average_trade_pl'], expected_stats['average_trade_pl'], places=4)
        self.assertEqual(result['total_trades'], expected_stats['total_trades'])

    def test_trade_stats_no_completed_trades(self) -> None:
        trade_log_only_buys: List[Dict[str, Any]] = [{'type': 'BUY', 'price': 100}]
        expected_stats = {'winning_trade_percentage': np.nan, 'average_trade_pl': np.nan, 'total_trades': 0.0}
        
        result = calculate_trade_stats(trade_log_only_buys)
        self.assertTrue(np.isnan(result['winning_trade_percentage']))
        self.assertTrue(np.isnan(result['average_trade_pl']))
        self.assertEqual(result['total_trades'], expected_stats['total_trades'])

        trade_log_buy_buy_sell: List[Dict[str, Any]] = [ # Only last BUY/SELL pair completes
            {'type': 'BUY', 'price': 100}, 
            {'type': 'BUY', 'price': 105}, {'type': 'SELL', 'price': 110} # P/L = +5
        ]
        result_2 = calculate_trade_stats(trade_log_buy_buy_sell)
        self.assertAlmostEqual(result_2['winning_trade_percentage'], 100.0, places=4)
        self.assertAlmostEqual(result_2['average_trade_pl'], 5.0, places=4)
        self.assertEqual(result_2['total_trades'], 1.0)


    def test_trade_stats_missing_price_in_log(self) -> None:
        trade_log: List[Dict[str, Any]] = [
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 110}, # Valid
            {'type': 'BUY', 'price': None}, {'type': 'SELL', 'price': 90},  # BUY price missing
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': None}, # SELL price missing
            {'type': 'BUY', 'price': 100}, {'type': 'SELL', 'price': 120}, # Valid
        ]
        # Expected: only first and last trades are completed
        # P/L: +10, +20. Avg P/L = 15. Win % = 100. Total = 2
        expected_stats = {'winning_trade_percentage': 100.0, 'average_trade_pl': 15.0, 'total_trades': 2.0}
        result = calculate_trade_stats(trade_log)
        self.assertAlmostEqual(result['winning_trade_percentage'], expected_stats['winning_trade_percentage'], places=4)
        self.assertAlmostEqual(result['average_trade_pl'], expected_stats['average_trade_pl'], places=4)
        self.assertEqual(result['total_trades'], expected_stats['total_trades'])

        trade_log_all_invalid: List[Dict[str, Any]] = [
            {'type': 'BUY', 'price': None}, {'type': 'SELL', 'price': None},
        ]
        expected_stats_all_invalid = {'winning_trade_percentage': np.nan, 'average_trade_pl': np.nan, 'total_trades': 0.0}
        result_all_invalid = calculate_trade_stats(trade_log_all_invalid)
        self.assertTrue(np.isnan(result_all_invalid['winning_trade_percentage']))
        self.assertTrue(np.isnan(result_all_invalid['average_trade_pl']))
        self.assertEqual(result_all_invalid['total_trades'], expected_stats_all_invalid['total_trades'])


if __name__ == '__main__':
    unittest.main()
