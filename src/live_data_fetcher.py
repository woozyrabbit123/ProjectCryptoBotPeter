import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Any
import configparser
import os
import logging

logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """
    A class responsible for fetching historical OHLCV data.
    It attempts to load data from a CSV path specified in config.ini,
    falling back to a hardcoded sample DataFrame if loading fails or
    config is not present.
    """

    def __init__(self) -> None:
        """
        Initializes the HistoricalDataFetcher.
        Reads 'config.ini' to get the path for mock OHLCV data.
        """
        self.mock_data_path: str = "data/mock_ohlcv_data.csv" # Default path
        
        config = configparser.ConfigParser()
        # Path to config.ini is relative to this file's location (src/live_data_fetcher.py)
        # So, ../../config.ini would point to the project root.
        config_file_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini') # Corrected path assuming src/ is one level from root

        if os.path.exists(config_file_path):
            config.read(config_file_path)
            if 'DataFetcher' in config and 'mock_data_path' in config['DataFetcher']:
                configured_path = config['DataFetcher']['mock_data_path']
                if configured_path: # Ensure it's not an empty string
                    self.mock_data_path = configured_path
                    logger.info(f"HistoricalDataFetcher: Using mock OHLCV data path from config.ini: {self.mock_data_path}")
                else:
                    logger.warning(f"HistoricalDataFetcher: 'mock_data_path' in config.ini is empty. Using default: {self.mock_data_path}")
            else:
                logger.warning(f"HistoricalDataFetcher: '[DataFetcher]' section or 'mock_data_path' key not found in {config_file_path}. Using default: {self.mock_data_path}")
        else:
            logger.warning(f"HistoricalDataFetcher: config.ini not found at {config_file_path}. Using default mock data path: {self.mock_data_path}")


    def _get_hardcoded_fallback_data(self) -> pd.DataFrame:
        """
        Returns the hardcoded sample DataFrame.
        """
        logger.info("HistoricalDataFetcher: Providing hardcoded fallback OHLCV data.")
        data: Dict[str, List[Any]] = {
            'timestamp': [
                "2023-01-01 00:00:00", "2023-01-01 00:01:00", "2023-01-01 00:02:00",
                "2023-01-01 00:03:00", "2023-01-01 00:04:00",
            ],
            'open': [100.0, 102.5, 100.0, 99.0, 100.0],
            'high': [105.0, 103.0, 101.5, 100.5, 102.0],
            'low': [98.0, 99.5, 98.5, 97.0, 99.0],
            'close': [102.5, 100.0, 99.0, 100.0, 101.0],
            'volume': [1000.0, 950.0, 1100.0, 1200.0, 1050.0],
        }
        df = pd.DataFrame(data)
        df['timestamp'] = df['timestamp'].astype(str)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def fetch_ohlcv_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical OHLCV data.
        Attempts to load from the path in self.mock_data_path (resolved from project root).
        Falls back to hardcoded data if loading fails.

        Args:
            symbol (str): The trading symbol (e.g., 'BTC/USD'). Ignored if loading mock data.
            start_date (str): The start date for the data. Ignored if loading mock data.
            end_date (str): The end date for the data. Ignored if loading mock data.

        Returns:
            pd.DataFrame: A DataFrame with OHLCV data.
        """
        # Parameters symbol, start_date, end_date are currently ignored by mock loading
        # but are kept for future API compatibility.
        _ = symbol
        _ = start_date
        _ = end_date

        path_to_load = self.mock_data_path
        if not os.path.isabs(path_to_load):
            # Assuming self.mock_data_path is relative to the project root.
            # __file__ is src/live_data_fetcher.py
            # Project root is one level up from 'src' directory.
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
            path_to_load = os.path.join(project_root, path_to_load)
        
        logger.info(f"HistoricalDataFetcher: Attempting to load mock OHLCV data from: {path_to_load}")
        
        try:
            df = pd.read_csv(path_to_load)
            if df.empty:
                logger.error(f"Mock OHLCV data file at {path_to_load} is empty. Falling back to hardcoded sample data.")
                return self._get_hardcoded_fallback_data()

            # Ensure correct dtypes (example: assuming CSV columns match hardcoded ones)
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Mock OHLCV data file at {path_to_load} is missing required columns. Expected: {required_columns}. Got: {list(df.columns)}. Falling back.")
                return self._get_hardcoded_fallback_data()

            df['timestamp'] = df['timestamp'].astype(str)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce ensures conversion, NaNs for errors
            
            # Check for NaNs introduced by coercion if strict adherence is needed
            if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
                logger.warning(f"NaN values found in numeric columns after coercion for {path_to_load}. Check data integrity.")

            logger.info(f"Successfully loaded mock OHLCV data from configured path: {path_to_load}")
            return df
        except FileNotFoundError:
            logger.error(f"Mock OHLCV data file not found at configured/resolved path: {path_to_load}. Falling back to hardcoded sample data.")
        except pd.errors.EmptyDataError: # Should be caught by df.empty above, but as a safeguard
            logger.error(f"Mock OHLCV data file at {path_to_load} is empty (pd.errors.EmptyDataError). Falling back to hardcoded sample data.")
        except Exception as e:
            logger.error(f"Error loading mock OHLCV data from {path_to_load}: {e}. Falling back to hardcoded sample data.", exc_info=True)
        
        return self._get_hardcoded_fallback_data()


if __name__ == '__main__':
    # Example usage:
    # Ensure basic logging for example run if not configured by main app
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fetcher = HistoricalDataFetcher()
    
    # Test with default config path (assuming data/mock_ohlcv_data.csv might exist or not)
    print(f"\n--- Attempting to load data (current mock_data_path: {fetcher.mock_data_path}) ---")
    sample_data_default = fetcher.fetch_ohlcv_data(
        symbol="BTC/USD",
        start_date="2023-01-01",
        end_date="2023-01-02"
    )
    print("\nFetched Data (default path):")
    print(sample_data_default.head())
    # sample_data_default.info()

    # Example: Simulate config.ini pointing to a non-existent file to test fallback
    print("\n--- Simulating config pointing to a non-existent file ---")
    fetcher.mock_data_path = "non_existent_data.csv" # Override path for this test
    sample_data_non_existent = fetcher.fetch_ohlcv_data(
        symbol="BTC/USD",
        start_date="2023-01-01",
        end_date="2023-01-02"
    )
    print("\nFetched Data (non-existent path - should be fallback):")
    print(sample_data_non_existent.head())
    # sample_data_non_existent.info()

    # To fully test, you would create a dummy config.ini and a dummy data/mock_ohlcv_data.csv
    # For instance, create a dummy 'temp_config.ini' and 'temp_data.csv'
    # and temporarily modify the fetcher's config_file_path for a test instance.
    # This is beyond a simple __main__ example.
