import logging
import random
import os
import sys
import signal
import time
import psutil
from src.logic_dna import LogicDNA, mutate_dna, LogicDNA_v1, ConditionNode, ActionNode, CompositeNode, SequenceNode
from utils.hardware_monitor import get_cpu_load
from src.nanostrat import run_nanostrat_test
from src.fitness import evaluate_dna_fitness
from src.experimental_pool import ExperimentalPoolManager
# from src.system_orchestrator import SystemOrchestrator  # Removed to avoid circular import
from settings import LEE_SETTINGS, EPM_SETTINGS, SO_SETTINGS, FERAL_CALIBRATOR_SETTINGS, LOGGING_SETTINGS, GENERAL_SETTINGS
from src.data_logger import log_event, logger_instance
from typing import List, Optional, Dict, Any, Tuple
from src.market_persona import MarketPersona
from typing import TYPE_CHECKING
import pandas as pd
import numpy as np
import csv
import json
from datetime import datetime

# Global state for MVP feedback
cycle_counter = 0
mutation_strength = LEE_SETTINGS['MUTATION_STRENGTH']

# Instantiate managers
epm = ExperimentalPoolManager()
# so = SystemOrchestrator()  # Removed to avoid circular import

# Defensive settings check utility
def check_settings_dict(settings_dict: Dict[str, Any], required_keys: List[str], dict_name: str) -> None:
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        msg = f"CRITICAL: Missing keys in {dict_name}: {missing}"
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        raise RuntimeError(msg)

# Defensive checks for all settings dicts
check_settings_dict(LEE_SETTINGS, ['MUTATION_STRENGTH', 'CYCLE_GENERATE_INTERVAL', 'DUMMY_MARKET_DATA_SIZE', 'MVL_BUFFER_SIZE'], 'LEE_SETTINGS')
check_settings_dict(EPM_SETTINGS, ['MAX_POOL_SIZE', 'MAX_POOL_LIFE', 'MIN_PERFORMANCE_THRESHOLD', 'MIN_POOL_SIZE', 'GRADUATION_MIN_SHARPE_POOL', 'GRADUATION_MIN_PNL_POOL', 'GRADUATION_MIN_TICKS_IN_POOL', 'GRADUATION_MIN_ACTIVATIONS_POOL', 'GRADUATION_EPSILON'], 'EPM_SETTINGS')
check_settings_dict(SO_SETTINGS, ['MAX_CANDIDATE_SLOTS', 'CANDIDATE_VIRTUAL_CAPITAL', 'CANDIDATE_MIN_SHARPE', 'CANDIDATE_MIN_PNL', 'CANDIDATE_MIN_TICKS', 'CANDIDATE_MIN_ACTIVATIONS', 'CANDIDATE_EPSILON', 'CANDIDATE_MAX_LIFE', 'CANDIDATE_SHARPE_DROP_TRIGGER', 'AB_TEST_TICKS', 'FITNESS_WEIGHTS', 'AB_SHARPE_THRESHOLD', 'AB_PNL_THRESHOLD', 'INFLUENCE_MULTIPLIER_MAX', 'INFLUENCE_MULTIPLIER_MIN', 'INFLUENCE_MULTIPLIER_STEP_UP', 'INFLUENCE_MULTIPLIER_STEP_DOWN', 'INFLUENCE_SCORE_HIGH_PCT', 'INFLUENCE_SCORE_LOW_PCT', 'COOLDOWN_TICKS_AFTER_TWEAK', 'KPI_DASHBOARD_INTERVAL'], 'SO_SETTINGS')

def get_dummy_market_data(n: int = LEE_SETTINGS['DUMMY_MARKET_DATA_SIZE']) -> List[Dict[str, float]]:
    """
    Generate dummy market data for testing.
    Args:
        n (int): Number of ticks to generate.
    Returns:
        list: List of dicts with price and RSI_14 values.
    """
    data: List[Dict[str, float]] = []
    price: float = 100.0
    for i in range(n):
        price += random.uniform(-1, 1)
        rsi: float = random.uniform(10, 90)
        data.append({'price': price, 'RSI_14': rsi})
    return data

def should_generate_new_dna() -> bool:
    """
    Determine if a new DNA should be generated this cycle.
    Returns:
        bool: True if new DNA should be generated.
    """
    global cycle_counter
    # Assuming 'so' is an instance of SystemOrchestrator, which might not be fully typed here due to potential circular deps
    # For now, we'll assume it has these attributes.
    sensitivity: int = so.meta_param_monitor.generation_trigger_sensitivity if so.meta_param_monitor.enabled else LEE_SETTINGS['GENERATION_TRIGGER_SENSITIVITY']
    return cycle_counter % sensitivity == 0 or epm.get_pool_size() < EPM_SETTINGS['MIN_POOL_SIZE']

def run_mvl_cycle() -> None:
    """
    Run a single MVL (main value loop) cycle: generate/test DNA, update pool, and log results.
    Adds error handling for external calls and calculations.
    Logs DNA_GENERATED_MVL event after each cycle.
    """
    global cycle_counter, mutation_strength
    logger = logging.getLogger(__name__) # Local logger instance
    cycle_counter += 1
    logger.info(f"=== MVL Cycle {cycle_counter} ===")
    if should_generate_new_dna():
        logger.info("Trigger: Generating new candidate DNA.")
        seed: LogicDNA = LogicDNA.seed_rsi_buy()
        candidate: LogicDNA = mutate_dna(seed, mutation_strength)
        cpu_load: float
        try:
            cpu_load = get_cpu_load()
        except Exception as e:
            logger.warning(f"LEE: Failed to get CPU load: {e}. Using safe default (99.0)")
            cpu_load = 99.0
        
        market_data: List[Dict[str, float]] = get_dummy_market_data(LEE_SETTINGS['DUMMY_MARKET_DATA_SIZE'])
        test_results: Dict[str, Any]
        try:
            test_results = run_nanostrat_test(candidate, market_data)
        except Exception as e:
            logger.warning(f"LEE: Error during nanostrat test: {e}. Using default test results.")
            test_results = {'virtual_pnl': 0.0, 'other_metrics': {}} # type: ignore
        
        fitness: str # evaluate_dna_fitness returns a string like 'discard_performance' or 'survived_mvl'
        try:
            fitness = evaluate_dna_fitness(test_results, cpu_load)
        except Exception as e:
            logger.warning(f"LEE: Error during fitness evaluation: {e}. Using 'discard_performance'.")
            fitness = 'discard_performance'
            
        logger.info(f"MVL Outcome: DNA={candidate} | TestResults={test_results} | CPU={cpu_load:.2f}% | Fitness={fitness}")
        # Structured logging for DNA_GENERATED_MVL
        log_event( # type: ignore
            'DNA_GENERATED_MVL', # type: ignore
            { # type: ignore
                'dna_id': candidate.id, # type: ignore
                'parent_id': candidate.parent_id, # type: ignore
                'initial_parameters': candidate.to_dict(), # type: ignore
                'mvl_test_pnl': test_results.get('virtual_pnl', None), # type: ignore
                'mvl_test_activations': test_results.get('activations', None), # type: ignore
                'mvl_cpu_load': cpu_load, # type: ignore
                'mvl_outcome': fitness, # type: ignore
            } # type: ignore
        ) # type: ignore
        # Optional: MVP feedback tweak
        if fitness == 'discard_performance':
            mutation_strength = max(0.05, mutation_strength * 0.95)
        elif fitness == 'survived_mvl':
            mutation_strength = min(0.2, mutation_strength * 1.05)
            epm.add_dna_to_pool(candidate) # type: ignore
            # Structured logging for EPM_DNA_ADDED
            log_event( # type: ignore
                'EPM_DNA_ADDED', # type: ignore
                { # type: ignore
                    'dna_id': candidate.id, # type: ignore
                    'initial_fitness_from_mvl': fitness, # type: ignore
                } # type: ignore
            ) # type: ignore
    else:
        logger.info("No new DNA generated this cycle.")

# --- RUN_CONFIGURATION_SNAPSHOT ---
def get_hardware_status() -> Dict[str, Any]:
    return {
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_load_percent': psutil.cpu_percent(interval=0.1),
        'virtual_memory': dict(psutil.virtual_memory()._asdict()),
    }

def check_log_file_accessible(log_file: str) -> Any: # Can be bool or str
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('')
        return True
    except Exception as e:
        return str(e)

# Compose config snapshot
def log_run_configuration_snapshot() -> None:
    log_file: str = GENERAL_SETTINGS['STRUCTURED_DATA_LOG_FILE']
    log_event('RUN_CONFIGURATION_SNAPSHOT', { # type: ignore
        'timestamp_start_run': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()), # type: ignore
        'bot_version': '0.1.0-dev', # type: ignore
        'LEE_SETTINGS': dict(LEE_SETTINGS), # type: ignore
        'EPM_SETTINGS': dict(EPM_SETTINGS), # type: ignore
        'SO_SETTINGS': dict(SO_SETTINGS), # type: ignore
        'FERAL_CALIBRATOR_SETTINGS': dict(FERAL_CALIBRATOR_SETTINGS) if 'FERAL_CALIBRATOR_SETTINGS' in globals() else {}, # type: ignore
        'LOGGING_SETTINGS': dict(LOGGING_SETTINGS), # type: ignore
        'GENERAL_SETTINGS': dict(GENERAL_SETTINGS), # type: ignore
        'META_PARAM_SETTINGS': {k: v for k, v in globals().items() if k.startswith('META_PARAM_') or k == 'ENABLE_META_SELF_TUNING'}, # type: ignore
        'hardware_status': get_hardware_status(), # type: ignore
        'log_file_accessible': check_log_file_accessible(log_file), # type: ignore
        'datalogger_initialized': isinstance(logger_instance, object), # type: ignore
    }) # type: ignore

# --- RUN_COMPLETED_SUMMARY ---
def log_run_completed_summary(start_time: float, dnas_generated: int, candidates_graduated: int, ab_tests: int, meta_param_events: int, final_pnl: float, critical_errors: List[str]) -> None:
    end_time: float = time.time()
    log_event('RUN_COMPLETED_SUMMARY', { # type: ignore
        'timestamp_end_run': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(end_time)), # type: ignore
        'run_duration_sec': end_time - start_time, # type: ignore
        'total_dnas_generated': dnas_generated, # type: ignore
        'total_candidates_graduated': candidates_graduated, # type: ignore
        'total_ab_tests': ab_tests, # type: ignore
        'total_meta_param_self_adjusted': meta_param_events, # type: ignore
        'final_portfolio_pnl': final_pnl, # type: ignore
        'critical_errors_encountered': critical_errors, # type: ignore
    }) # type: ignore

# --- Signal handler for graceful shutdown ---
shutdown_requested: bool = False
def handle_sigint(sig: int, frame: Any) -> None:
    global shutdown_requested
    shutdown_requested = True
    print("\nSIGINT received. Preparing graceful shutdown...")

signal.signal(signal.SIGINT, handle_sigint)

# --- Main logic ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log_run_configuration_snapshot()
    start_time = time.time()
    dnas_generated = 0
    candidates_graduated = 0
    ab_tests = 0
    meta_param_events = 0
    critical_errors = []
    prev_price = 100.0
    market_data_buffer = []
    BUFFER_SIZE = LEE_SETTINGS['MVL_BUFFER_SIZE']
    try:
        for i in range(40):
            if shutdown_requested:
                break
            run_mvl_cycle()
            dnas_generated += 1  # Approximate, refine if needed
            price = prev_price + random.uniform(-1, 1)
            rsi = random.uniform(10, 90)
            market_tick = {'price': price, 'price_prev': prev_price, 'RSI_14': rsi}
            epm.evaluate_pool_tick(market_tick)
            prev_price = price
            market_data_buffer.append({'price': price, 'price_prev': prev_price, 'RSI_14': rsi})
            if len(market_data_buffer) > BUFFER_SIZE:
                market_data_buffer.pop(0)
            if i % 5 == 0:
                epm.prune_pool()
            graduates = epm.check_for_graduates()
            for idx, entry in reversed(graduates):
                grad_entry = epm.pop_graduate(idx)
                so.promote_to_candidate_slot(grad_entry)
                candidates_graduated += 1
                log_event(
                    'CANDIDATE_GRADUATED',
                    {
                        'dna_id': grad_entry.dna.id,
                        'epm_performance_summary': {
                            'rolling_pnl_pool': grad_entry.rolling_pnl_pool,
                            'activation_count_pool': grad_entry.activation_count_pool,
                            'ticks_in_pool': grad_entry.ticks_in_pool,
                        },
                        'candidate_slot_id_assigned': grad_entry.dna.id,
                    }
                )
            so.evaluate_candidates_per_tick(market_tick, recent_market_data=market_data_buffer)
            logging.info(f"SO State: NumCandidates={len(so.candidate_slots)} | Candidates={[str(slot.dna) for slot in so.candidate_slots]}")
            boost = so.get_candidate_confidence_boost('buy')
            logging.info(f"SO Confidence Boost (buy): {boost}")
            logging.info(f"EPM State: PoolSize={epm.get_pool_size()} | AvgPoolPnL={epm.get_average_pool_pnl():.2f}")
            so.log_portfolio_metrics()
        # Gather run summary stats
        ab_tests = getattr(so, 'ab_tests_completed', 0)
        meta_param_events = getattr(so.meta_param_monitor, 'window_counter', 0)
        final_pnl = sum(slot.pnl_candidate for slot in so.candidate_slots)
        if hasattr(so, 'critical_errors'):
            critical_errors = so.critical_errors[-5:]
    except Exception as e:
        critical_errors.append(str(e))
        log_event('CRITICAL_ERROR', {'exception': str(e)})
    finally:
        log_run_completed_summary(start_time, dnas_generated, candidates_graduated, ab_tests, meta_param_events, final_pnl, critical_errors)
        print("Graceful shutdown complete. Run summary logged.")

class LEE:
    """
    Logic Evolution Engine (LEE) for evolving LogicDNA_v1 populations.
    Handles initialization, population management, and (eventually) evolutionary cycles.
    """
    def __init__(self,
                 population_size: int,
                 mutation_rate_parametric: float,
                 mutation_rate_structural: float,
                 crossover_rate: float,
                 elitism_percentage: float,
                 random_injection_percentage: float,
                 max_depth: int,
                 max_nodes: int,
                 complexity_weights: Dict[str, float]):
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.population_size: int = population_size
        self.mutation_rate_parametric: float = mutation_rate_parametric
        self.mutation_rate_structural: float = mutation_rate_structural
        self.crossover_rate: float = crossover_rate
        self.elitism_percentage: float = elitism_percentage
        self.random_injection_percentage: float = random_injection_percentage
        self.max_depth: int = max_depth
        self.max_nodes: int = max_nodes
        self.complexity_weights: Dict[str, float] = complexity_weights.copy() if complexity_weights else {}
        self.population: List[LogicDNA_v1] = []
        self.generation_counter: int = 0
        self.mle_motif_seeding_influence_factor: float = 0.15
        self.AVAILABLE_INDICATORS: Dict[str, List[int]] = {
            'SMA': [10, 20, 50],
            'EMA': [10, 20, 50],
            'RSI': [14],
            'VOLATILITY': [20]  # This one causes the alias issue
        }
        self.performance_log_path: str = "performance_log.csv" # Default, can be updated by SO

    def initialize_population(self, seed_dna_templates: Optional[List[LogicDNA_v1]] = None) -> None:
        """
        Initialize the population using hybrid seeding (some from templates, some random).
        Set generation_born=0 for all new individuals.
        """
        self.population = []
        n_seed: int = int(0.2 * self.population_size) if seed_dna_templates else 0
        n_random: int = self.population_size - n_seed
        # Seeded individuals
        if seed_dna_templates:
            for _ in range(n_seed):
                template: LogicDNA_v1 = random.choice(seed_dna_templates)
                # Copy and (optionally) mutate slightly (for now, just copy)
                dna: LogicDNA_v1 = template.copy()
                dna.generation_born = 0
                self.population.append(dna)
        # Random individuals
        for _ in range(n_random):
            tree: LogicDNA_v1 = self._create_random_valid_tree(generation_born=0)
            self.population.append(tree)

    def _create_random_valid_tree(self, generation_born: Optional[int] = None, mle_bias: Optional[Dict[str, Any]] = None) -> LogicDNA_v1:
        """
        Generate a random valid LogicDNA_v1 tree respecting max_depth, max_nodes, and node type rules.
        If mle_bias/seed_motifs is present, with probability self.mle_motif_seeding_influence_factor, seed a motif.
        """
        # Motif seeding logic
        motifs: List[Tuple[str, Any]] = [] # type: ignore
        if mle_bias and 'seed_motifs' in mle_bias and mle_bias['seed_motifs']:
            motifs = sorted(mle_bias['seed_motifs'].items(), key=lambda x: -x[1]) # type: ignore
        if motifs and random.random() < self.mle_motif_seeding_influence_factor:
            motif_str, _ = random.choice(motifs[:min(5, len(motifs))])
            if motif_str.startswith('Indicator_') and '_Used' in motif_str:
                try:
                    _, ind, period_str, _ = motif_str.split('_')
                    period: int = int(period_str)
                    node: ConditionNode = ConditionNode(
                        indicator_id=ind,
                        comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                        threshold_value=random.uniform(5, 100),
                        lookback_period_1=period,
                        lookback_period_2=None
                    )
                    tree: LogicDNA_v1 = LogicDNA_v1(root_node=node, generation_born=generation_born)
                    if tree.is_valid(self.max_depth, self.max_nodes):
                        return tree
                except Exception: # Broad exception to catch parsing errors, etc.
                    pass  # Fallback to normal random
            # TODO: For more complex motifs (Condition-Action, Composite), parse and build tree segment
            # For now, fallback to normal random if not a simple indicator motif

        def _random_node(depth: int) -> Any: # Returns LogicNode but LogicNode is ABC
            if depth >= self.max_depth:
                return ActionNode(action_type=random.choice(["BUY", "SELL", "HOLD"]), size_factor=random.uniform(0.0, 1.0))
            
            node_type: str = random.choice(["Condition", "Action", "Composite", "Sequence"])
            if node_type == "Action":
                return ActionNode(action_type=random.choice(["BUY", "SELL", "HOLD"]), size_factor=random.uniform(0.0, 1.0))
            elif node_type == "Condition":
                indicator_id: str = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
                lookback_period_1: int
                if indicator_id == 'RSI':
                    lookback_period_1 = 14
                elif indicator_id == 'VOLATILITY':
                    lookback_period_1 = 20
                else:
                    lookback_period_1 = random.choice(self.AVAILABLE_INDICATORS[indicator_id])
                return ConditionNode(
                    indicator_id=indicator_id,
                    comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                    threshold_value=random.uniform(5, 100),
                    lookback_period_1=lookback_period_1,
                    lookback_period_2=None # Assuming no secondary lookback for now
                )
            elif node_type == "Composite":
                return CompositeNode(
                    logical_operator=random.choice(["AND", "OR"]),
                    child1=_random_node(depth + 1),
                    child2=_random_node(depth + 1)
                )
            elif node_type == "Sequence":
                return SequenceNode(
                    child1=_random_node(depth + 1),
                    child2=_random_node(depth + 1)
                )
            else: # Fallback
                return ActionNode(action_type="HOLD", size_factor=0.0)

        for _ in range(10):  # Try up to 10 times to get a valid tree
            root_node: Any = _random_node(1) # LogicNode
            tree = LogicDNA_v1(root_node=root_node, generation_born=generation_born)
            if tree.is_valid(self.max_depth, self.max_nodes):
                return tree
        # Fallback if no valid tree generated after attempts
        return LogicDNA_v1(root_node=ActionNode(action_type="HOLD", size_factor=0.0), generation_born=generation_born)

    def _run_backtest_for_dna(self, dna: LogicDNA_v1, historical_ohlcv: pd.DataFrame, precomputed_indicators: pd.DataFrame, backtest_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest a LogicDNA_v1 individual on historical data.
        Simulates a simple portfolio with buy/sell/hold actions, applies transaction costs, and logs trades.
        Returns a dict of performance metrics.
        """
        initial_capital: float = backtest_config.get('initial_capital', 10000.0)
        transaction_cost_pct: float = backtest_config.get('transaction_cost_pct', 0.001)
        cash: float = initial_capital
        position: float = 0.0
        position_entry_price: float = 0.0
        equity_curve: List[float] = []
        trade_log: List[Dict[str, Any]] = []
        trade_count: int = 0
        # last_action: str = 'HOLD' # Currently unused

        for i in range(len(historical_ohlcv)):
            row: pd.Series = historical_ohlcv.iloc[i]
            indicators_row: pd.Series = precomputed_indicators.iloc[i]
            available_indicators: Dict[str, Any] = indicators_row.to_dict()

            if 'realized_vol_20' in available_indicators:
                available_indicators['VOLATILITY_20'] = available_indicators['realized_vol_20']
            
            market_state: Dict[str, Any] = row.to_dict()
            # ts = row['timestamp'] if 'timestamp' in row else i # Currently unused
            action_signal: Any = dna.evaluate(market_state, available_indicators) # Can be bool or tuple
            
            price: float = row['close']
            if np.isnan(price) or any(np.isnan(v) for v in available_indicators.values() if isinstance(v, (int, float))):
                current_equity = cash + (position * price if not np.isnan(price) else 0)
                equity_curve.append(current_equity if not np.isnan(current_equity) else cash)
                continue

            action: str
            size_factor: float
            if isinstance(action_signal, tuple) and len(action_signal) == 2:
                action, size_factor = action_signal
            else: # Assuming boolean result if not tuple
                action, size_factor = ("HOLD", 0.0)

            if action == 'BUY' and position == 0:
                max_affordable: float = cash / (price * (1 + transaction_cost_pct))
                size: float = max_affordable * size_factor
                cost: float = size * price * (1 + transaction_cost_pct)
                if cost <= cash and size > 0:
                    cash -= cost
                    position = size
                    position_entry_price = price
                    trade_ts_val: Any = row['timestamp']
                    trade_ts_str: str = trade_ts_val.isoformat() if hasattr(trade_ts_val, 'isoformat') else str(trade_ts_val)
                    trade_log.append({'timestamp': trade_ts_str, 'type': 'BUY', 'price': float(price), 'size': float(size)})
                    trade_count += 1
                    # last_action = 'BUY'
            elif action == 'SELL' and position > 0:
                proceeds: float = position * price * (1 - transaction_cost_pct)
                cash += proceeds
                trade_ts_val: Any = row['timestamp']
                trade_ts_str: str = trade_ts_val.isoformat() if hasattr(trade_ts_val, 'isoformat') else str(trade_ts_val)
                trade_log.append({'timestamp': trade_ts_str, 'type': 'SELL', 'price': float(price), 'size': float(position)})
                position = 0.0
                # position_entry_price = 0.0 # Resetting, though not strictly necessary here
                trade_count += 1
                # last_action = 'SELL'
            
            current_equity = cash + (position * price if not np.isnan(price) else 0)
            equity_curve.append(current_equity if not np.isnan(current_equity) else cash)

        final_equity: float = cash + (position * historical_ohlcv.iloc[-1]['close'] if not historical_ohlcv.empty and not np.isnan(historical_ohlcv.iloc[-1]['close']) else 0)
        
        returns: np.ndarray = np.array([])
        if len(equity_curve) > 1:
            eq_array = np.array(equity_curve)
            # Ensure no zero or negative values before division if returns are relative
            # For simplicity, direct diff used, but filter out NaNs/Infs from returns
            returns = np.diff(eq_array) / eq_array[:-1]
            returns = returns[np.isfinite(returns)]


        profit: float = final_equity - initial_capital
        sharpe: float = (np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)) if len(returns) > 2 else 0.0
        
        max_drawdown: float = 0.0
        peak: float = equity_curve[0] if equity_curve else initial_capital
        for eq_val in equity_curve:
            if eq_val > peak:
                peak = eq_val
            dd: float = (peak - eq_val) / peak if peak != 0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd
        
        metrics: Dict[str, Any] = {
            'metric_profit': float(profit),
            'metric_sharpe_ratio': float(sharpe),
            'metric_max_drawdown': float(max_drawdown),
            'metric_trade_count': int(trade_count),
            'metric_final_equity': float(final_equity),
            'metric_initial_capital': float(initial_capital),
            'trade_log': trade_log,
        }
        return metrics

    def _evaluate_population(self, active_persona: 'MarketPersona', market_data_for_evaluation: Dict[str, Any]) -> None:
        """
        Evaluate each LogicDNA_v1 in the population using the real backtester.
        Expects market_data_for_evaluation to be a dict with keys:
            'ohlcv': pd.DataFrame
            'indicators': pd.DataFrame
            'backtest_config': Dict[str, Any]
            'active_persona_name': str
            'ces_vector': Dict[str, Any]
            'performance_log_path': str
            'current_generation': int
        Logs performance for MLE after each evaluation.
        """
        overall_start_time: float = time.time()

        # Data Preparation Timing
        prep_start_time: float = time.time()
        ohlcv: pd.DataFrame = market_data_for_evaluation['ohlcv']
        indicators: pd.DataFrame = market_data_for_evaluation['indicators']
        backtest_config: Dict[str, Any] = market_data_for_evaluation['backtest_config']
        active_persona_name: str = market_data_for_evaluation.get('active_persona_name', 'UNKNOWN')
        ces_vector: Dict[str, Any] = market_data_for_evaluation.get('ces_vector', {})
        performance_log_path: str = market_data_for_evaluation.get('performance_log_path', self.performance_log_path)
        current_generation: int = market_data_for_evaluation.get('current_generation', getattr(self, 'generation_counter', 0))
        prep_end_time: float = time.time()
        self.logger.info(f"Time taken for data preparation: {prep_end_time - prep_start_time:.3f} seconds")

        # Prepare log file
        log_exists: bool = os.path.exists(performance_log_path)
        with open(performance_log_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames: List[str] = [
                'dna_id', 'generation_born', 'current_generation_evaluated',
                'logic_dna_structure_representation', 'performance_metrics',
                'fitness_score', 'active_persona_name', 'ces_vector_at_evaluation_time', 'timestamp_of_evaluation'
            ]
            writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not log_exists:
                writer.writeheader()

            for dna_idx, dna_individual in enumerate(self.population):
                dna_id_str: str = getattr(dna_individual, 'dna_id', f"unknown_dna_{dna_idx}")
                self.logger.debug(f"Processing DNA {dna_id_str} ({dna_idx + 1}/{len(self.population)})")

                # Backtest and Fitness Calculation Timing
                backtest_fitness_start_time: float = time.time()
                performance_metrics: Dict[str, Any] = self._run_backtest_for_dna(dna_individual, ohlcv, indicators, backtest_config)
                
                node_count: int
                tree_depth: int
                node_count, tree_depth = dna_individual.calculate_complexity()
                
                complexity_score: float = (
                    self.complexity_weights.get('nodes', 1.0) * node_count +
                    self.complexity_weights.get('depth', 1.0) * tree_depth
                )
                current_fitness: float = active_persona.calculate_fitness(performance_metrics, complexity_score)
                dna_individual.fitness = current_fitness # type: ignore
                setattr(dna_individual, 'performance_metrics', performance_metrics) # Optionally store for logging
                backtest_fitness_end_time: float = time.time()
                self.logger.debug(f"Time taken for DNA {dna_id_str} backtest and fitness calculation: {backtest_fitness_end_time - backtest_fitness_start_time:.3f} seconds")

                structure_repr_str: str = ''
                if hasattr(dna_individual, 'to_string_representation') and callable(getattr(dna_individual, 'to_string_representation')):
                    structure_repr_str = dna_individual.to_string_representation()

                # CSV Log Writing Timing
                log_write_start_time: float = time.time()
                writer.writerow({
                    'dna_id': dna_id_str,
                    'generation_born': getattr(dna_individual, 'generation_born', ''),
                    'current_generation_evaluated': current_generation,
                    'logic_dna_structure_representation': structure_repr_str,
                    'performance_metrics': json.dumps(performance_metrics),
                    'fitness_score': current_fitness,
                    'active_persona_name': active_persona_name,
                    'ces_vector_at_evaluation_time': json.dumps(ces_vector),
                    'timestamp_of_evaluation': datetime.now().isoformat()
                })
                log_write_end_time: float = time.time()
                self.logger.debug(f"Time taken for DNA {dna_id_str} performance log entry: {log_write_end_time - log_write_start_time:.3f} seconds")

        overall_end_time: float = time.time()
        self.logger.info(f"Total time taken for _evaluate_population: {overall_end_time - overall_start_time:.3f} seconds")
    # FIXME: Resolve VOLATILITY_20 alias situation (see _run_backtest_for_dna method)

    def _select_parents(self) -> Tuple[List[LogicDNA_v1], List[LogicDNA_v1]]:
        """
        Select elites and parents for reproduction using elitism and tournament selection.
        Returns (elites, parents_for_reproduction)
        """
        # Sort by fitness (descending)
        sorted_pop: List[LogicDNA_v1] = sorted(self.population, key=lambda d: getattr(d, 'fitness', 0.0), reverse=True)
        n_elite: int = max(1, int(self.elitism_percentage * self.population_size))
        elites: List[LogicDNA_v1] = [dna.copy() for dna in sorted_pop[:n_elite]]
        
        # Tournament selection for parents
        parents_for_reproduction: List[LogicDNA_v1] = []
        tournament_size: int = 3
        while len(parents_for_reproduction) < self.population_size:
            tournament_candidates: List[LogicDNA_v1] = random.sample(sorted_pop, min(tournament_size, len(sorted_pop)))
            winner: LogicDNA_v1 = max(tournament_candidates, key=lambda d: getattr(d, 'fitness', 0.0))
            parents_for_reproduction.append(winner.copy())
        return elites, parents_for_reproduction

    def run_evolutionary_cycle(self, active_persona: 'MarketPersona', market_data_for_evaluation: Dict[str, Any], mle_bias: Optional[Dict[str, Any]] = None, ces_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Run one full evolutionary cycle: evaluate, select, reproduce, inject randoms, update population.
        Accepts optional mle_bias and ces_info for feedback integration.
        """
        orig_mutation_rate_structural: float = self.mutation_rate_structural
        orig_mutation_rate_parametric: float = self.mutation_rate_parametric
        
        if ces_info is not None and isinstance(ces_info, dict):
            v: float = ces_info.get('volatility', 0.5)
            l: float = ces_info.get('liquidity', 0.5)
            t: float = ces_info.get('trend', 0.5)
            composite_stress: float = (v + (1 - l) + t) / 3.0
            adj: float = 0.05 if composite_stress > 0.7 else (-0.03 if composite_stress < 0.4 else 0.0)
            self.mutation_rate_structural = min(1.0, max(0.01, orig_mutation_rate_structural + adj))
            self.mutation_rate_parametric = min(1.0, max(0.01, orig_mutation_rate_parametric + adj))
            self.logger.info(f"[LEE] Adjusted mutation rates for CES: structural={self.mutation_rate_structural:.3f}, parametric={self.mutation_rate_parametric:.3f} (composite_stress={composite_stress:.2f})")

        if mle_bias is not None:
            self.logger.info(f"[LEE] MLE Bias (seed_motifs): {mle_bias.get('seed_motifs', {})}")
            self.logger.info(f"[LEE] MLE Bias (operator_biases): {mle_bias.get('recommended_operator_biases', {})}")
        if ces_info is not None:
            self.logger.info(f"[LEE] CES Info: {ces_info}")
            
        self._evaluate_population(active_persona, market_data_for_evaluation)
        self.generation_counter += 1
        
        fitnesses: List[float] = [getattr(dna, 'fitness', 0.0) for dna in self.population]
        best_fitness: Optional[float] = max(fitnesses) if fitnesses else None
        avg_fitness: Optional[float] = sum(fitnesses) / len(fitnesses) if fitnesses else None
        self.logger.info(f"Generation: {self.generation_counter} - Best Fitness: {best_fitness:.3f if best_fitness is not None else 'N/A'}, Avg Fitness: {avg_fitness:.3f if avg_fitness is not None else 'N/A'}")
        
        elites: List[LogicDNA_v1]
        parents: List[LogicDNA_v1]
        elites, parents = self._select_parents()
        
        num_random_injections: int = int(self.random_injection_percentage * self.population_size)
        num_offspring_needed: int = self.population_size - len(elites) - num_random_injections
        
        offspring: List[LogicDNA_v1] = self._reproduce(parents, num_offspring_needed, mle_bias)
        random_new_individuals: List[LogicDNA_v1] = [self._create_random_valid_tree(mle_bias=mle_bias, generation_born=self.generation_counter) for _ in range(num_random_injections)]
        
        self.population = elites + offspring + random_new_individuals
        
        # --- Diversity metric ---
        unique_structures: set[str] = set()
        for dna_indiv in self.population:
            if hasattr(dna_indiv, 'to_string_representation'):
                unique_structures.add(dna_indiv.to_string_representation())
        diversity: int = len(unique_structures)
        diversity_pct: float = diversity / self.population_size if self.population_size > 0 else 0.0
        self.logger.info(f"[LEE] Population diversity: {diversity} unique structures ({diversity_pct:.2%})")
        log_event('POP_DIVERSITY', {'generation': self.generation_counter, 'diversity': diversity, 'diversity_pct': diversity_pct}) # type: ignore
        
        if not hasattr(self, '_diversity_low_counter'):
            self._diversity_low_counter: int = 0
        if diversity_pct < 0.2:
            self._diversity_low_counter += 1
            if self._diversity_low_counter >= 3:
                self.logger.warning("WARNING: Population Diversity Low!")
                log_event('DIVERSITY_COLLAPSE_WARNING', {'generation': self.generation_counter, 'diversity': diversity, 'diversity_pct': diversity_pct}) # type: ignore
        else:
            self._diversity_low_counter = 0
            
        self.mutation_rate_structural = orig_mutation_rate_structural
        self.mutation_rate_parametric = orig_mutation_rate_parametric
        self.logger.info(f"LEE Generation: Best fitness={best_fitness}, Avg fitness={avg_fitness}")

    def _reproduce(self, parents_for_reproduction: List[LogicDNA_v1], num_offspring_needed: int, mle_bias: Optional[Dict[str, Any]] = None) -> List[LogicDNA_v1]:
        """
        Generate offspring using crossover and mutation.
        Accepts optional mle_bias for motif/operator biasing (placeholder logic).
        """
        offspring: List[LogicDNA_v1] = []
        current_gen: int = getattr(self, 'generation_counter', 0)
        motif_adopted_count: int = 0
        
        for _ in range(num_offspring_needed):
            child_candidate: LogicDNA_v1
            parent1: LogicDNA_v1
            parent2: LogicDNA_v1
            
            if random.random() < self.crossover_rate and len(parents_for_reproduction) > 1:
                parent1, parent2 = random.sample(parents_for_reproduction, 2)
                child1, child2 = self._crossover_trees(parent1, parent2)
                child_candidate = child1 if random.random() < 0.5 else child2
            else:
                child_candidate = random.choice(parents_for_reproduction).copy()
            
            is_motif_adopted_this_iteration: bool = False
            if random.random() < self.mutation_rate_structural:
                original_structure: str = child_candidate.to_string_representation() if hasattr(child_candidate, 'to_string_representation') else ''
                child_candidate = self._structural_mutation(child_candidate, mle_bias=mle_bias)
                new_structure: str = child_candidate.to_string_representation() if hasattr(child_candidate, 'to_string_representation') else ''
                if mle_bias and 'seed_motifs' in mle_bias and original_structure != new_structure:
                    is_motif_adopted_this_iteration = True # Heuristic
                    
            child_candidate = self._parametric_mutation(child_candidate) # Parametric mutation always applied
            
            if child_candidate.is_valid(self.max_depth, self.max_nodes):
                child_candidate.generation_born = current_gen
                offspring.append(child_candidate)
            else:
                # Fallback to a new random tree if mutated/crossed-over child is invalid
                new_random_tree: LogicDNA_v1 = self._create_random_valid_tree(generation_born=current_gen, mle_bias=mle_bias)
                offspring.append(new_random_tree)
                
            if is_motif_adopted_this_iteration:
                motif_adopted_count +=1
                
        self.logger.info(f"[LEE] Motif adoption rate this generation: {motif_adopted_count}/{num_offspring_needed if num_offspring_needed > 0 else 'N/A'}")
        log_event('MOTIF_ADOPTION', {'generation': current_gen, 'motif_adopted': motif_adopted_count, 'num_offspring': num_offspring_needed}) # type: ignore
        return offspring

    def _parametric_mutation(self, dna_individual: LogicDNA_v1) -> LogicDNA_v1:
        """
        Mutate numerical parameters of nodes in a copy of the DNA.
        For ConditionNodes, only mutate to valid indicator/period combinations.
        """
        dna_copy: LogicDNA_v1 = dna_individual.copy()
        
        def _mutate_node_recursive(node: Any) -> None: # Node can be LogicNode
            if isinstance(node, ConditionNode):
                if random.random() < 0.2: # Chance to change indicator
                    new_indicator_id: str = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
                    node.indicator_id = new_indicator_id
                    if new_indicator_id == 'RSI':
                        node.lookback_period_1 = 14
                    elif new_indicator_id == 'VOLATILITY':
                        node.lookback_period_1 = 20
                    else:
                        node.lookback_period_1 = random.choice(self.AVAILABLE_INDICATORS[new_indicator_id])
                else: # Chance to change only lookback period if indicator is not RSI/VOLATILITY
                    if node.indicator_id not in ['RSI', 'VOLATILITY'] and random.random() < 0.5 :
                         node.lookback_period_1 = random.choice(self.AVAILABLE_INDICATORS[node.indicator_id])
                
                if random.random() < 0.5: # Chance to mutate threshold
                    node.threshold_value += random.uniform(-2.0, 2.0) # Small adjustment
                    node.threshold_value = round(node.threshold_value, 4) # Keep it tidy

            elif isinstance(node, ActionNode):
                if random.random() < 0.5: # Chance to mutate size_factor
                    node.size_factor = min(1.0, max(0.0, node.size_factor + random.uniform(-0.1, 0.1)))
                    node.size_factor = round(node.size_factor, 4)

            elif isinstance(node, (CompositeNode, SequenceNode)):
                if node.child1:
                    _mutate_node_recursive(node.child1)
                if node.child2:
                    _mutate_node_recursive(node.child2)

        if dna_copy.root_node:
            _mutate_node_recursive(dna_copy.root_node)
        return dna_copy

    def _structural_mutation(self, dna_individual: LogicDNA_v1, mle_bias: Optional[Dict[str, Any]] = None) -> LogicDNA_v1:
        """
        Randomly add, prune, or replace a node/subtree in a copy of the DNA.
        If mle_bias/seed_motifs is present, with probability, inject a motif node/branch.
        """
        dna_copy: LogicDNA_v1 = dna_individual.copy()
        
        effective_motif_inject_prob: float = 0.15
        if mle_bias and 'recommended_operator_biases' in mle_bias:
            effective_motif_inject_prob = mle_bias['recommended_operator_biases'].get('structural_mutation_rate_adjustment_factor', 0.15)
        
        motifs_data: List[Tuple[str, Any]] = [] # type: ignore
        if mle_bias and 'seed_motifs' in mle_bias and mle_bias['seed_motifs']:
            motifs_data = sorted(mle_bias['seed_motifs'].items(), key=lambda x: -x[1]) # type: ignore

        if motifs_data and random.random() < effective_motif_inject_prob:
            motif_str, _ = random.choice(motifs_data[:min(5, len(motifs_data))])
            if motif_str.startswith('Indicator_') and '_Used' in motif_str:
                try:
                    _, ind, period_str, _ = motif_str.split('_')
                    period: int = int(period_str)
                    new_node_from_motif: ConditionNode = ConditionNode(
                        indicator_id=ind,
                        comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                        threshold_value=random.uniform(5, 100),
                        lookback_period_1=period
                    )
                    # add_node_at_random_valid_point returns bool, not handling it here
                    dna_copy.add_node_at_random_valid_point(new_node_from_motif)
                    if dna_copy.is_valid(self.max_depth, self.max_nodes):
                         return dna_copy
                except Exception: # Fallback to normal mutation
                    pass
            # TODO: For more complex motifs, parse and inject as subtree

        # Fallback to normal structural mutation
        mutation_type: str = random.choice(['add', 'prune', 'replace'])
        
        if mutation_type == 'add':
            new_indicator_id: str = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
            new_lookback: int
            if new_indicator_id == 'RSI': new_lookback = 14
            elif new_indicator_id == 'VOLATILITY': new_lookback = 20
            else: new_lookback = random.choice(self.AVAILABLE_INDICATORS[new_indicator_id])
            
            node_to_add: ConditionNode = ConditionNode(
                indicator_id=new_indicator_id,
                comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                threshold_value=random.uniform(5, 100),
                lookback_period_1=new_lookback
            )
            dna_copy.add_node_at_random_valid_point(node_to_add)
            
        elif mutation_type == 'prune':
            dna_copy.prune_random_subtree() # This method might make the tree invalid if root is pruned to None implicitly
            if dna_copy.root_node is None: # Ensure root is not None
                 dna_copy.root_node = ActionNode(action_type="HOLD", size_factor=0.0)


        elif mutation_type == 'replace':
            node_to_replace_obj: Any = dna_copy.get_random_node() # LogicNode
            if node_to_replace_obj and hasattr(node_to_replace_obj, 'node_id'):
                replacement_node: Any # LogicNode
                if random.random() < 0.5:
                    replacement_node = ActionNode(action_type="HOLD", size_factor=0.0)
                else:
                    repl_indicator_id: str = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
                    repl_lookback: int
                    if repl_indicator_id == 'RSI': repl_lookback = 14
                    elif repl_indicator_id == 'VOLATILITY': repl_lookback = 20
                    else: repl_lookback = random.choice(self.AVAILABLE_INDICATORS[repl_indicator_id])
                    replacement_node = ConditionNode(
                        indicator_id=repl_indicator_id,
                        comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                        threshold_value=random.uniform(5, 100),
                        lookback_period_1=repl_lookback
                    )
                dna_copy.replace_node(node_to_replace_obj.node_id, replacement_node)
        
        # Ensure validity after mutation
        if not dna_copy.is_valid(self.max_depth, self.max_nodes):
            # If mutation results in invalid tree, return original or a new random one
            # For simplicity, returning copy of original to avoid breaking evolution with too many randoms
            return dna_individual.copy() 
        return dna_copy

    def _crossover_trees(self, parent1: LogicDNA_v1, parent2: LogicDNA_v1) -> Tuple[LogicDNA_v1, LogicDNA_v1]:
        """
        Single-point subtree exchange between two parents.
        """
        child1: LogicDNA_v1 = parent1.copy()
        child2: LogicDNA_v1 = parent2.copy()
        
        # Select random points (nodes) in each parent tree
        # LogicNode is an ABC, so using Any for node types from get_random_subtree_point
        node1_to_swap: Optional[Any] = child1.get_random_subtree_point() 
        node2_to_swap: Optional[Any] = child2.get_random_subtree_point()

        if node1_to_swap and node2_to_swap and hasattr(node1_to_swap, 'node_id') and hasattr(node2_to_swap, 'node_id'):
            # Perform the swap: node1_id in child1 gets node2_copy, node2_id in child2 gets node1_copy
            # Need to copy the nodes before replacing to avoid object aliasing issues if nodes are complex
            node1_id_str: str = node1_to_swap.node_id
            node2_id_str: str = node2_to_swap.node_id
            
            node1_subtree_copy_for_child2: Any = node1_to_swap.copy()
            node2_subtree_copy_for_child1: Any = node2_to_swap.copy()

            child1.replace_node(node1_id_str, node2_subtree_copy_for_child1)
            child2.replace_node(node2_id_str, node1_subtree_copy_for_child2)

        # Ensure validity of children after crossover
        current_gen: int = getattr(self, 'generation_counter', 0)
        if not child1.is_valid(self.max_depth, self.max_nodes):
            child1 = self._create_random_valid_tree(generation_born=current_gen) # Fallback
        if not child2.is_valid(self.max_depth, self.max_nodes):
            child2 = self._create_random_valid_tree(generation_born=current_gen) # Fallback
            
        return child1, child2