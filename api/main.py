import logging
import os
import pandas as pd
import json
import numpy as np
from flask import Flask, jsonify, request
from typing import Dict, Any, List, Optional, Tuple
import configparser # For reading config.ini

# --- Logging Setup ---
try:
    from src.utils.logging_utils import setup_global_logging
    setup_global_logging('config.ini')
    # print("API: Successfully configured global logging from src.utils.")
except ImportError:
    # print("API Warning: src.utils.logging_utils not found. Using basic logging for API.")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [API] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

app = Flask(__name__)
logger = logging.getLogger(__name__)

# --- Helper Functions (Adapted from scripts.generate_simple_performance_report) ---

def calculate_sharpe_ratio(equity_curve: List[float], trading_days_per_year: int = 252) -> float:
    if not equity_curve or len(equity_curve) < 2:
        return np.nan
    equity_array = np.array(equity_curve, dtype=float)
    returns = np.diff(equity_array) / equity_array[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return np.nan
    mean_return = np.mean(returns)
    std_dev_returns = np.std(returns)
    if std_dev_returns == 0:
        return np.nan
    sharpe = (mean_return / std_dev_returns) * np.sqrt(trading_days_per_year)
    return float(sharpe)

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    if not equity_curve:
        return np.nan
    equity_array = np.array(equity_curve, dtype=float)
    peak = -np.inf
    max_dd = 0.0
    for equity_value in equity_array:
        if equity_value > peak:
            peak = equity_value
        if peak > 0:
            drawdown = (peak - equity_value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    return float(max_dd * 100)

def calculate_trade_stats(trade_log: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trade_log:
        return {'winning_trade_percentage': np.nan, 'average_trade_pl': np.nan, 'total_trades': 0}
    trades_completed: List[Dict[str, float]] = []
    active_buy: Optional[Dict[str, Any]] = None
    for trade_action in trade_log:
        action_type = trade_action.get('type')
        price = trade_action.get('price')
        if price is None:
            continue
        if action_type == 'BUY':
            active_buy = trade_action
        elif action_type == 'SELL' and active_buy:
            buy_price = active_buy.get('price')
            if buy_price is not None:
                profit_loss = price - buy_price
                trades_completed.append({'pnl': profit_loss, 'buy_price': buy_price, 'sell_price': price})
            active_buy = None
    if not trades_completed:
        return {'winning_trade_percentage': np.nan, 'average_trade_pl': np.nan, 'total_trades': 0}
    winning_trades = sum(1 for trade in trades_completed if trade['pnl'] > 0)
    total_trades = len(trades_completed)
    winning_percentage = (winning_trades / total_trades) * 100 if total_trades > 0 else np.nan
    average_pl = sum(trade['pnl'] for trade in trades_completed) / total_trades if total_trades > 0 else np.nan
    return {'winning_trade_percentage': float(winning_percentage), 'average_trade_pl': float(average_pl), 'total_trades': total_trades}

def process_log_data_for_api(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Processes the DataFrame to calculate statistics for the API response.
    """
    api_stats: Dict[str, Any] = {
        'highest_fitness': np.nan,
        'avg_fitness_per_generation': {},
        'total_unique_dna_structures': 0,
        'average_sharpe_ratio': np.nan,
        'average_max_drawdown': np.nan,
        'average_winning_trade_percentage': np.nan,
        'average_trade_pl_overall': np.nan,
        'dna_with_trades_count': 0,
        'evaluated_dna_count': len(df)
    }

    # Basic stats
    if 'fitness_score' in df.columns:
        df['fitness_score_numeric'] = pd.to_numeric(df['fitness_score'], errors='coerce')
        api_stats['highest_fitness'] = df['fitness_score_numeric'].max()
        valid_fitness_df = df.dropna(subset=['fitness_score_numeric'])
        if not valid_fitness_df.empty:
            api_stats['avg_fitness_per_generation'] = valid_fitness_df.groupby('current_generation_evaluated')['fitness_score_numeric'].mean().to_dict()

    if 'logic_dna_structure_representation' in df.columns:
        api_stats['total_unique_dna_structures'] = df['logic_dna_structure_representation'].nunique()

    # Metrics from JSON 'performance_metrics'
    individual_sharpes: List[float] = []
    individual_max_dds: List[float] = []
    individual_win_rates: List[float] = []
    individual_avg_pls: List[float] = []

    for index, row in df.iterrows():
        metrics_json_str = row.get('performance_metrics')
        if pd.isna(metrics_json_str):
            continue
        try:
            perf_metrics_dict = json.loads(metrics_json_str)
        except json.JSONDecodeError:
            logger.warning(f"Row {index}: Failed to parse performance_metrics JSON. Skipping. Content: '{metrics_json_str[:100]}...'")
            continue

        equity_curve = perf_metrics_dict.get('equity_curve')
        trade_log = perf_metrics_dict.get('trade_log')

        if isinstance(equity_curve, list) and equity_curve:
            sharpe = calculate_sharpe_ratio(equity_curve)
            if not np.isnan(sharpe): individual_sharpes.append(sharpe)
            max_dd = calculate_max_drawdown(equity_curve)
            if not np.isnan(max_dd): individual_max_dds.append(max_dd)
        
        if isinstance(trade_log, list): # trade_stats handles empty list
            trade_stats = calculate_trade_stats(trade_log)
            if trade_stats['total_trades'] > 0:
                api_stats['dna_with_trades_count'] += 1
                if not np.isnan(trade_stats['winning_trade_percentage']): individual_win_rates.append(trade_stats['winning_trade_percentage'])
                if not np.isnan(trade_stats['average_trade_pl']): individual_avg_pls.append(trade_stats['average_trade_pl'])

    if individual_sharpes: api_stats['average_sharpe_ratio'] = np.mean(individual_sharpes)
    if individual_max_dds: api_stats['average_max_drawdown'] = np.mean(individual_max_dds)
    if individual_win_rates: api_stats['average_winning_trade_percentage'] = np.mean(individual_win_rates)
    if individual_avg_pls: api_stats['average_trade_pl_overall'] = np.mean(individual_avg_pls)
    
    # Clean up NaN to None for JSON compatibility if preferred
    for key, value in api_stats.items():
        if isinstance(value, float) and np.isnan(value):
            api_stats[key] = None 
        elif isinstance(value, dict): # For avg_fitness_per_generation
            api_stats[key] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in value.items()}


    return api_stats

# --- Flask Endpoint ---

@app.route('/api/performance_metrics', methods=['GET'])
def get_performance_metrics() -> Tuple[str, int]:
    logger.info("Received request for /api/performance_metrics")
    log_file_arg: Optional[str] = request.args.get('log_file_path')
    
    log_file_name: str = log_file_arg if log_file_arg else "performance_log_FULL_V1_2.csv"
    actual_log_file_path: str = log_file_name

    # Path resolution logic (simplified from script for API context)
    # Assumes API might be run from project root, or path provided is absolute/resolvable
    if not os.path.isabs(actual_log_file_path) and not os.path.exists(actual_log_file_path):
        # Try path relative to /app (common in sandbox/container)
        path_in_app: str = os.path.join("/app", log_file_name)
        if os.path.exists(path_in_app):
            actual_log_file_path = path_in_app
    
    logger.info(f"Attempting to read performance log from: {actual_log_file_path}")

    try:
        performance_df: pd.DataFrame = pd.read_csv(actual_log_file_path)
        if performance_df.empty:
            logger.warning(f"Log file '{actual_log_file_path}' is empty.")
            return jsonify({'error': f"Log file is empty: {log_file_name}"}), 404
        logger.info(f"Successfully read log file '{actual_log_file_path}'. Shape: {performance_df.shape}")
    except FileNotFoundError:
        logger.error(f"Log file not found at {actual_log_file_path}.")
        return jsonify({'error': f"Log file not found: {log_file_name}"}), 404
    except pd.errors.EmptyDataError: # Should be caught by df.empty check, but good practice
        logger.error(f"Log file at {actual_log_file_path} is empty (EmptyDataError).")
        return jsonify({'error': f"Log file is empty: {log_file_name}"}), 404
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading log file '{actual_log_file_path}': {e}", exc_info=True)
        return jsonify({'error': f"Error reading log file: {str(e)}"}), 500

    try:
        logger.info("Processing performance data for API...")
        summary_stats: Dict[str, Any] = process_log_data_for_api(performance_df)
        response_data = {"log_file": os.path.basename(actual_log_file_path), **summary_stats}
        logger.info(f"Successfully processed data. Returning metrics for {log_file_name}.")
        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"Error processing data for API from log file '{actual_log_file_path}': {e}", exc_info=True)
        return jsonify({'error': f"Error processing data: {str(e)}"}), 500


# --- Main Block ---
if __name__ == '__main__':
    api_host: str = "0.0.0.0"
    api_port: int = 5000

    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini') # Assumes config.ini is in project root

    try:
        if os.path.exists(config_path):
            config.read(config_path)
            if 'APISettings' in config:
                api_host = config['APISettings'].get('host', api_host)
                api_port = config['APISettings'].getint('port', api_port)
                logger.info(f"Loaded API settings from config.ini: Host={api_host}, Port={api_port}")
            else:
                logger.info("No [APISettings] section in config.ini, using defaults.")
        else:
            logger.info("config.ini not found, using default API host and port.")
    except Exception as e:
        logger.error(f"Error reading API settings from config.ini: {e}. Using defaults.", exc_info=True)

    logger.info(f"Starting Flask API server on {api_host}:{api_port}")
    app.run(host=api_host, port=api_port, debug=False)
