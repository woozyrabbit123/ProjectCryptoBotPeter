import numpy as np
import pandas as pd # Though not directly used by the functions themselves, good for context or future growth
from typing import List, Dict, Any, Optional 
import logging # For any potential logging within these utils, though not strictly needed by current functions

logger = logging.getLogger(__name__)

# --- Metric Calculation Functions ---

def calculate_sharpe_ratio(equity_curve: List[float], trading_days_per_year: int = 252) -> float:
    """
    Calculates Sharpe ratio from an equity curve.

    Args:
        equity_curve (List[float]): A list representing equity values over time.
        trading_days_per_year (int): Number of trading days in a year for annualization.

    Returns:
        float: The calculated Sharpe ratio, or np.nan if calculation is not possible.
    """
    if not equity_curve or len(equity_curve) < 2:
        return np.nan
    
    equity_array = np.array(equity_curve, dtype=float)
    # Calculate daily returns: (current_equity - previous_equity) / previous_equity
    # Using percentage returns
    returns = np.diff(equity_array) / equity_array[:-1]
    returns = returns[np.isfinite(returns)] # Remove NaNs/Infs that might arise from division by zero

    if len(returns) < 2: # Need at least 2 returns to calculate std dev
        return np.nan

    mean_return = np.mean(returns)
    std_dev_returns = np.std(returns)

    if std_dev_returns == 0:
        # If std_dev is 0, Sharpe ratio is undefined or infinite.
        # Return np.nan for consistency, or handle as per specific financial logic (e.g., large positive if mean_return > 0).
        return np.nan 

    sharpe = (mean_return / std_dev_returns) * np.sqrt(trading_days_per_year)
    return float(sharpe)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculates the maximum drawdown from an equity curve.

    Args:
        equity_curve (List[float]): A list representing equity values over time.

    Returns:
        float: The maximum drawdown as a percentage (e.g., 15.5 for 15.5%), 
               or np.nan if calculation is not possible.
    """
    if not equity_curve:
        return np.nan
    
    equity_array = np.array(equity_curve, dtype=float)
    peak = -np.inf # Initialize peak to negative infinity
    max_dd = 0.0

    for equity_value in equity_array:
        if equity_value > peak:
            peak = equity_value
        # Ensure peak is not zero to avoid division by zero if equity can drop to or below zero
        # Also, drawdown is typically positive, so ensure peak is positive if values are positive.
        if peak > 0: 
            drawdown = (peak - equity_value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    return float(max_dd * 100) # Return as percentage


def calculate_trade_stats(trade_log: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates winning trade percentage, average Profit/Loss (P/L) per unit,
    and total number of completed trades. Assumes trades are pairs of 'BUY' 
    and a subsequent 'SELL'.

    Args:
        trade_log (List[Dict[str, Any]]): A list of trade action dictionaries.
                                          Each dict should have 'type' (BUY/SELL) 
                                          and 'price'.

    Returns:
        Dict[str, float]: A dictionary containing:
                          'winning_trade_percentage' (float, or np.nan),
                          'average_trade_pl' (float, or np.nan),
                          'total_trades' (int, cast to float for consistency here, or can be int).
                          np.nan is used if metrics cannot be calculated (e.g., no trades).
    """
    if not trade_log:
        return {'winning_trade_percentage': np.nan, 'average_trade_pl': np.nan, 'total_trades': 0.0}

    trades_completed: List[Dict[str, float]] = []
    active_buy: Optional[Dict[str, Any]] = None

    for trade_action in trade_log:
        action_type = trade_action.get('type')
        price = trade_action.get('price')

        if price is None: 
            # Using logger from the module if these functions were ever to log independently
            # logger.debug(f"Skipping trade action due to missing price: {trade_action}") 
            continue # Skip if price is missing

        if action_type == 'BUY':
            # If there's an active buy without a sell, this new BUY overwrites it.
            # This handles cases like multiple BUYs before a SELL.
            active_buy = trade_action
        elif action_type == 'SELL' and active_buy:
            buy_price = active_buy.get('price')
            # buy_size = active_buy.get('size') # Size not used for per-unit P/L
            if buy_price is not None: # Ensure buy_price was valid
                profit_loss = price - buy_price # Per unit P/L
                trades_completed.append({'pnl': profit_loss, 'buy_price': buy_price, 'sell_price': price})
            active_buy = None # Reset active buy after a sell completes a trade

    if not trades_completed:
        return {'winning_trade_percentage': np.nan, 'average_trade_pl': np.nan, 'total_trades': 0.0}

    winning_trades = sum(1 for trade in trades_completed if trade['pnl'] > 0)
    total_trades = len(trades_completed)
    
    winning_percentage = (winning_trades / total_trades) * 100 if total_trades > 0 else np.nan
    average_pl = sum(trade['pnl'] for trade in trades_completed) / total_trades if total_trades > 0 else np.nan
    
    return {
        'winning_trade_percentage': float(winning_percentage), 
        'average_trade_pl': float(average_pl), 
        'total_trades': float(total_trades) # Cast to float for Dict[str, float] consistency
    }
