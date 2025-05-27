"""
Shard logging module for Project Crypto Bot Peter.
Handles efficient binary logging of trading data using struct packing.
"""

import struct
from datetime import datetime, timezone
from typing import Optional
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# New Wisdom Shard format:
# <QfffBBfhBfffHBh
# Fields:
#   timestamp (uint64)
#   price (float32)
#   ema_price (float32)
#   calculated_volatility (float32)
#   volatility_regime (uint8)
#   action_taken (uint8)
#   pnl_realized (float32)
#   scaled_micro_momentum (int16)
#   volume_divergence_flag (uint8)
#   live_volume (float32)
#   vwma_price (float32)
#   vwma_deviation_pct (float32)
#   fetch_latency_ms (uint16)
#   latency_spike_flag (uint8)
#   latency_volatility_index (float16)
SHARD_FORMAT = '<QfffBBfhBfffHBh'  # 37 bytes

def log_raw_shard(
    file_path: str,
    timestamp_int: int,
    price: float,
    ema_price: float,
    calculated_volatility: float,
    volatility_regime: int,
    action_taken: int,
    pnl_realized: float,
    micro_momentum: float,
    volume_divergence_flag: int = 0,
    live_volume: float = 0.0,
    vwma_price: float = 0.0,
    vwma_deviation_pct: float = 0.0,
    fetch_latency_ms: int = 0,
    latency_spike_flag: int = 0,
    latency_volatility_index: float = 0.0
) -> None:
    """
    Log a single 'wisdom shard' record to a binary file in a compact 37-byte format.
    Fields:
        timestamp (uint64)
        price (float32)
        ema_price (float32)
        calculated_volatility (float32)
        volatility_regime (uint8)
        action_taken (uint8)
        pnl_realized (float32)
        micro_momentum_scaled (int16, scaled by 10000)
        volume_divergence_flag (uint8)
        live_volume (float32)
        vwma_price (float32)
        vwma_deviation_pct (float32)
        fetch_latency_ms (uint16)
        latency_spike_flag (uint8)
        latency_volatility_index (float16)
    Args:
        file_path (str): Path to the binary log file.
        timestamp_int (int): Timestamp as seconds since epoch (uint64).
        price (float): The price at the time of logging.
        ema_price (float): The EMA price at the time of logging.
        calculated_volatility (float): Calculated volatility value (float32).
        volatility_regime (int): Volatility regime (uint8).
        action_taken (int): Action taken (uint8).
        pnl_realized (float): Realized PnL (float32).
        micro_momentum (float): Micro momentum (will be scaled and stored as int16).
        volume_divergence_flag (int): Volume divergence flag (uint8, default 0).
        live_volume (float): Live volume (float32, default 0.0).
        vwma_price (float): VWMA price (float32, default 0.0).
        vwma_deviation_pct (float): VWMA deviation percent (float32, default 0.0).
        fetch_latency_ms (int): Fetch latency in milliseconds (uint16, default 0).
        latency_spike_flag (int): Latency spike flag (uint8, default 0).
        latency_volatility_index (float): Latency volatility index (float16, default 0.0).
    """
    try:
        # Prepare all values for struct.pack and print their types/values for debugging
        ts = int(timestamp_int)
        pr = float(price)
        ema = float(ema_price)
        vol = float(calculated_volatility)
        regime = int(volatility_regime)
        action = int(action_taken)
        pnl = float(pnl_realized)
        momentum_scaled = int(max(-32767, min(32767, int(micro_momentum * 10000))))
        vol_div_flag = int(volume_divergence_flag)
        live_vol = float(live_volume)
        vwma = float(vwma_price)
        vwma_dev = float(vwma_deviation_pct)
        latency_ms = int(fetch_latency_ms)
        latency_spike = int(latency_spike_flag)
        latency_vol_idx = np.float16(latency_volatility_index) if latency_volatility_index is not None else np.float16(0.0)
        latency_vol_idx_int = int(latency_vol_idx)
        # Debug prints for all integer fields
        print(f"DEBUG: ts type: {type(ts)}, value: {ts}")
        print(f"DEBUG: regime type: {type(regime)}, value: {regime}")
        print(f"DEBUG: action type: {type(action)}, value: {action}")
        print(f"DEBUG: momentum_scaled type: {type(momentum_scaled)}, value: {momentum_scaled}")
        print(f"DEBUG: vol_div_flag type: {type(vol_div_flag)}, value: {vol_div_flag}")
        print(f"DEBUG: latency_ms type: {type(latency_ms)}, value: {latency_ms}")
        print(f"DEBUG: latency_spike type: {type(latency_spike)}, value: {latency_spike}")
        print(f"DEBUG: latency_vol_idx_int type: {type(latency_vol_idx_int)}, value: {latency_vol_idx_int}")
        packed = struct.pack(
            SHARD_FORMAT,
            ts,
            pr,
            ema,
            vol,
            regime,
            action,
            pnl,
            momentum_scaled,
            vol_div_flag,
            live_vol,
            vwma,
            vwma_dev,
            latency_ms,
            latency_spike,
            latency_vol_idx_int
        )
        with open(file_path, 'ab') as f:
            f.write(packed)
        logger.debug(f"Logged shard: ts={ts}, price={pr}, ema={ema}, vol={vol}, regime={regime}, action={action}, pnl={pnl}, momentum={momentum_scaled}, vol_div_flag={vol_div_flag}, live_vol={live_vol}, vwma={vwma}, vwma_dev={vwma_dev}, latency_ms={latency_ms}, latency_spike={latency_spike}, latency_vol_idx={latency_vol_idx}")
    except (IOError, struct.error, TypeError, ValueError) as e:
        logger.error(f"Error logging shard to {file_path}: {e}")

# TODO: Implement shard logging functions 