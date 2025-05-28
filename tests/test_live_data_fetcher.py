import unittest
import pandas as pd
from pandas import DataFrame
import numpy as np # For dtype checking

# Attempt to import the class to be tested
try:
    from src.live_data_fetcher import HistoricalDataFetcher
except ImportError:
    # This allows the test to be discovered and run even if the main module has issues
    # that are not directly related to the class being tested here, or if there's a PYTHONPATH issue.
    # The actual tests for the class methods will fail if the class cannot be imported.
    HistoricalDataFetcher = None 
    print("Warning: HistoricalDataFetcher could not be imported from src.live_data_fetcher. Tests may fail.")


class TestHistoricalDataFetcher(unittest.TestCase):
    """
    Test suite for the HistoricalDataFetcher class.
    """

    def setUp(self) -> None:
        """
        Set up for test methods. This method will be called before every test.
        """
        if HistoricalDataFetcher is not None:
            self.fetcher = HistoricalDataFetcher()
        else:
            # Fail fast if the class couldn't be imported, to make it clear.
            self.fail("HistoricalDataFetcher class not imported. Cannot run tests.")

    def test_fetch_ohlcv_data_exists(self) -> None:
        """
        Test if the fetch_ohlcv_data method exists on the fetcher instance.
        """
        self.assertTrue(hasattr(self.fetcher, 'fetch_ohlcv_data'), "fetch_ohlcv_data method should exist.")
        self.assertTrue(callable(self.fetcher.fetch_ohlcv_data), "fetch_ohlcv_data should be callable.")

    def test_fetch_ohlcv_data_returns_dataframe(self) -> None:
        """
        Test if calling fetch_ohlcv_data returns a pandas DataFrame.
        """
        # Using dummy parameters as the current implementation ignores them
        result = self.fetcher.fetch_ohlcv_data(symbol="ANY/SYM", start_date="2023-01-01", end_date="2023-01-02")
        self.assertIsInstance(result, pd.DataFrame, "fetch_ohlcv_data should return a pandas DataFrame.")

    def test_fetch_ohlcv_data_dataframe_structure(self) -> None:
        """
        Test if the returned DataFrame has the correct column names.
        """
        result_df = self.fetcher.fetch_ohlcv_data(symbol="ANY/SYM", start_date="2023-01-01", end_date="2023-01-02")
        expected_columns: List[str] = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(result_df.columns), expected_columns,
                             f"DataFrame columns should be {expected_columns}, but got {list(result_df.columns)}.")

    def test_fetch_ohlcv_data_dataframe_dtypes(self) -> None:
        """
        Test if the DataFrame columns have reasonable data types.
        'timestamp' should be object (string).
        'open', 'high', 'low', 'close', 'volume' should be numeric (float).
        """
        result_df = self.fetcher.fetch_ohlcv_data(symbol="ANY/SYM", start_date="2023-01-01", end_date="2023-01-02")
        
        # Check timestamp dtype
        # The hardcoded data explicitly sets it as string, which pandas often infers as 'object'
        self.assertEqual(result_df['timestamp'].dtype, object, 
                         f"Dtype for 'timestamp' should be object/string, but got {result_df['timestamp'].dtype}.")

        # Check numeric dtypes
        numeric_columns: List[str] = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result_df[col]),
                            f"Dtype for column '{col}' should be numeric, but got {result_df[col].dtype}.")
            # More specifically, the hardcoded data sets them to float
            self.assertTrue(pd.api.types.is_float_dtype(result_df[col]),
                            f"Dtype for column '{col}' should be float, but got {result_df[col].dtype}.")


    def test_fetch_ohlcv_data_not_empty(self) -> None:
        """
        Test if the returned DataFrame is not empty.
        """
        result_df = self.fetcher.fetch_ohlcv_data(symbol="ANY/SYM", start_date="2023-01-01", end_date="2023-01-02")
        self.assertFalse(result_df.empty, "The returned DataFrame should not be empty.")
        self.assertGreater(len(result_df), 0, "The returned DataFrame should have more than 0 rows.")


if __name__ == '__main__':
    unittest.main()
