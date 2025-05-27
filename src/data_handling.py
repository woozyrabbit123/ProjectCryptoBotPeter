"""
Data handling module for Project Crypto Bot Peter.
Handles market data loading and processing using Polars for efficient memory usage.
"""

import polars as pl
from typing import Optional
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_market_data_from_parquet(file_path: str) -> Optional[pl.LazyFrame]:
    """
    Load market data from a Parquet file into a Polars LazyFrame.
    
    Args:
        file_path (str): Path to the Parquet file containing market data.
        
    Returns:
        Optional[pl.LazyFrame]: A Polars LazyFrame containing the processed market data,
            or None if the file cannot be found or loaded.
            
    The function:
    - Loads data using polars.scan_parquet
    - Selects and processes specific columns
    - Casts numeric columns to Float32
    - Ensures timestamp is parsed as Datetime
    """
    try:
        # Verify file exists
        if not Path(file_path).exists():
            logger.error(f"Market data file not found: {file_path}")
            return None
            
        # Load data (removed memory_map=True for compatibility)
        lf = pl.scan_parquet(file_path)
        
        # Select and process columns
        lf = lf.select([
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("pair"),
            pl.col("price").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
            pl.col("sma_5").cast(pl.Float32)
        ])
        
        return lf
        
    except Exception as e:
        logger.error(f"Error loading market data from {file_path}: {str(e)}")
        return None

def calculate_simple_ema_polars(
    lf: pl.LazyFrame,
    period: int = 5,
    price_col: str = "price"
) -> Optional[pl.LazyFrame]:
    """
    Calculate Exponential Moving Average (EMA) for a given price column in a Polars LazyFrame.
    
    Args:
        lf (pl.LazyFrame): Input LazyFrame containing market data
        period (int, optional): EMA period. Defaults to 5.
        price_col (str, optional): Name of the price column to calculate EMA for. Defaults to "price".
        
    Returns:
        Optional[pl.LazyFrame]: LazyFrame with added EMA column, or None if input validation fails.
        
    Raises:
        ValueError: If period is not positive or if price_col doesn't exist in the schema
    """
    try:
        # Input validation
        if period <= 0:
            raise ValueError(f"EMA period must be positive, got {period}")
            
        # Check if price_col exists in schema
        schema = lf.collect_schema()
        if price_col not in schema:
            raise ValueError(f"Column '{price_col}' not found in LazyFrame schema")
            
        # Calculate EMA using ewm_mean with adjust=False for recursive formula
        ema_col = f"ema_{period}"
        alpha = 2 / (period + 1)  # Standard EMA smoothing factor
        
        lf = lf.with_columns([
            pl.col(price_col)
            .ewm_mean(alpha=alpha, adjust=False)
            .alias(ema_col)
        ])
        
        return lf
        
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return None

# TODO: Implement load_market_data_from_parquet and calculate_simple_ema_polars functions 

# --- Indicator Calculation Utilities ---
def load_ohlcv_csv(filepath):
    """
    Load OHLCV data from a CSV file into a pandas DataFrame.
    Expects columns: timestamp, open, high, low, close, volume
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df

def calculate_indicators(df):
    """
    Calculate standard indicators and return as a DataFrame (aligned with df).
    Indicators: SMA(10,20,50), EMA(10,20,50), RSI(14), realized volatility(20)
    """
    out = pd.DataFrame(index=df.index)
    # SMA
    for period in [10, 20, 50]:
        out[f'SMA_{period}'] = df['close'].rolling(window=period, min_periods=period).mean()
    # EMA
    for period in [10, 20, 50]:
        out[f'EMA_{period}'] = df['close'].ewm(span=period, min_periods=period, adjust=False).mean()
    # RSI (manual implementation)
    period = 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / (loss + 1e-10)
    out['RSI_14'] = 100 - (100 / (1 + rs))
    # Realized Volatility (rolling std of log returns)
    log_ret = np.log(df['close'] / df['close'].shift(1))
    out['realized_vol_20'] = log_ret.rolling(window=20, min_periods=20).std()
    return out 