"""
Tests for data_handling module functions.
"""

import pytest
import polars as pl
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_handling import load_market_data_from_parquet, calculate_simple_ema_polars

# Sample data for testing
SAMPLE_DATA = {
    "timestamp": [
        datetime(2024, 1, 1, 12, 0) + timedelta(minutes=i)
        for i in range(10)
    ],
    "pair": ["BTC/USD"] * 10,
    "price": [100.0 + i for i in range(10)],  # Simple increasing prices
    "volume": [1000.0 + i * 100 for i in range(10)],
    "sma_5": [100.0 + i * 0.8 for i in range(10)]  # Simulated SMA values
}

@pytest.fixture
def sample_parquet_file():
    """Create a temporary Parquet file with sample data for testing."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
    # Now write Parquet after closing the file
    df = pl.DataFrame(SAMPLE_DATA)
    df.write_parquet(tmp_path)
    yield tmp_path
    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)

@pytest.fixture
def sample_lazyframe():
    """Create a sample LazyFrame for testing EMA calculation."""
    return pl.LazyFrame(SAMPLE_DATA)

def test_load_market_data_from_parquet_success(sample_parquet_file):
    """Test successful loading of market data from Parquet file."""
    lf = load_market_data_from_parquet(sample_parquet_file)
    
    assert lf is not None, "LazyFrame should not be None for valid file"
    
    # Collect the data to verify contents
    df = lf.collect()
    
    # Check columns and dtypes
    assert df.schema["timestamp"] == pl.Datetime
    assert df.schema["price"] == pl.Float32
    assert df.schema["volume"] == pl.Float32
    assert df.schema["sma_5"] == pl.Float32
    
    # Verify data integrity
    assert len(df) == len(SAMPLE_DATA["timestamp"])
    assert df["pair"].unique().to_list() == ["BTC/USD"]
    assert df["price"].to_list() == SAMPLE_DATA["price"]

def test_load_market_data_from_parquet_file_not_found():
    """Test handling of non-existent file."""
    lf = load_market_data_from_parquet("nonexistent_file.parquet")
    assert lf is None, "Should return None for non-existent file"

@patch("polars.scan_parquet")
def test_load_market_data_from_parquet_scan_error(mock_scan_parquet):
    """Test handling of scan_parquet errors."""
    mock_scan_parquet.side_effect = Exception("Simulated scan error")
    lf = load_market_data_from_parquet("dummy.parquet")
    assert lf is None, "Should return None when scan_parquet raises an error"

def test_calculate_simple_ema_polars_success(sample_lazyframe):
    """Test successful EMA calculation."""
    period = 3
    lf = calculate_simple_ema_polars(sample_lazyframe, period=period)
    
    assert lf is not None, "LazyFrame should not be None for valid input"
    
    # Collect the data to verify EMA calculation
    df = lf.collect()
    
    # Verify EMA column exists
    ema_col = f"ema_{period}"
    assert ema_col in df.columns
    
    # Calculate expected EMA values manually for verification
    # For period=3, alpha=0.5, the EMA formula is:
    # EMA = price * alpha + previous_EMA * (1-alpha)
    expected_ema = []
    alpha = 2 / (period + 1)
    for i, price in enumerate(SAMPLE_DATA["price"]):
        if i == 0:
            expected_ema.append(price)  # First value is the price itself
        else:
            ema = price * alpha + expected_ema[-1] * (1 - alpha)
            expected_ema.append(ema)
    
    # Compare calculated EMA with expected values
    calculated_ema = df[ema_col].to_list()
    print("\nExpected EMA vs Actual EMA:")
    for i, (calc, expected) in enumerate(zip(calculated_ema, expected_ema)):
        print(f"Row {i}: Expected={expected}, Actual={calc}")
    # Allow for small floating-point differences
    for calc, expected in zip(calculated_ema, expected_ema):
        assert abs(calc - expected) < 1e-6, "EMA calculation mismatch"

def test_calculate_simple_ema_polars_invalid_period(sample_lazyframe):
    """Test EMA calculation with invalid period."""
    lf = calculate_simple_ema_polars(sample_lazyframe, period=0)
    assert lf is None, "Should return None for invalid period"

def test_calculate_simple_ema_polars_invalid_column(sample_lazyframe):
    """Test EMA calculation with non-existent price column."""
    lf = calculate_simple_ema_polars(sample_lazyframe, price_col="nonexistent_column")
    assert lf is None, "Should return None for non-existent price column"

def test_calculate_simple_ema_polars_custom_price_column(sample_lazyframe):
    """Test EMA calculation with a custom price column."""
    # Add a custom price column
    lf = sample_lazyframe.with_columns([
        (pl.col("price") * 1.1).alias("custom_price")
    ])
    
    # Calculate EMA on custom price column
    result = calculate_simple_ema_polars(lf, price_col="custom_price")
    assert result is not None, "Should handle custom price column"
    
    # Verify the EMA was calculated on the custom price
    df = result.collect()
    assert "ema_5" in df.columns
    assert df["ema_5"].mean() > df["price"].mean(), "EMA should reflect custom price values" 