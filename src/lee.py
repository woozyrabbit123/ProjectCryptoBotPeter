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
from typing import List, Optional
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
def check_settings_dict(settings_dict, required_keys, dict_name):
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        msg = f"CRITICAL: Missing keys in {dict_name}: {missing}"
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        raise RuntimeError(msg)

# Defensive checks for all settings dicts
check_settings_dict(LEE_SETTINGS, ['MUTATION_STRENGTH', 'CYCLE_GENERATE_INTERVAL', 'DUMMY_MARKET_DATA_SIZE', 'MVL_BUFFER_SIZE'], 'LEE_SETTINGS')
check_settings_dict(EPM_SETTINGS, ['MAX_POOL_SIZE', 'MAX_POOL_LIFE', 'MIN_PERFORMANCE_THRESHOLD', 'MIN_POOL_SIZE', 'GRADUATION_MIN_SHARPE_POOL', 'GRADUATION_MIN_PNL_POOL', 'GRADUATION_MIN_TICKS_IN_POOL', 'GRADUATION_MIN_ACTIVATIONS_POOL', 'GRADUATION_EPSILON'], 'EPM_SETTINGS')
check_settings_dict(SO_SETTINGS, ['MAX_CANDIDATE_SLOTS', 'CANDIDATE_VIRTUAL_CAPITAL', 'CANDIDATE_MIN_SHARPE', 'CANDIDATE_MIN_PNL', 'CANDIDATE_MIN_TICKS', 'CANDIDATE_MIN_ACTIVATIONS', 'CANDIDATE_EPSILON', 'CANDIDATE_MAX_LIFE', 'CANDIDATE_SHARPE_DROP_TRIGGER', 'AB_TEST_TICKS', 'FITNESS_WEIGHTS', 'AB_SHARPE_THRESHOLD', 'AB_PNL_THRESHOLD', 'INFLUENCE_MULTIPLIER_MAX', 'INFLUENCE_MULTIPLIER_MIN', 'INFLUENCE_MULTIPLIER_STEP_UP', 'INFLUENCE_MULTIPLIER_STEP_DOWN', 'INFLUENCE_SCORE_HIGH_PCT', 'INFLUENCE_SCORE_LOW_PCT', 'COOLDOWN_TICKS_AFTER_TWEAK', 'KPI_DASHBOARD_INTERVAL'], 'SO_SETTINGS')

def get_dummy_market_data(n=LEE_SETTINGS['DUMMY_MARKET_DATA_SIZE']):
    """
    Generate dummy market data for testing.
    Args:
        n (int): Number of ticks to generate.
    Returns:
        list: List of dicts with price and RSI_14 values.
    """
    data = []
    price = 100.0
    for i in range(n):
        price += random.uniform(-1, 1)
        rsi = random.uniform(10, 90)
        data.append({'price': price, 'RSI_14': rsi})
    return data

def should_generate_new_dna():
    """
    Determine if a new DNA should be generated this cycle.
    Returns:
        bool: True if new DNA should be generated.
    """
    global cycle_counter
    sensitivity = so.meta_param_monitor.generation_trigger_sensitivity if so.meta_param_monitor.enabled else LEE_SETTINGS['GENERATION_TRIGGER_SENSITIVITY']
    return cycle_counter % sensitivity == 0 or epm.get_pool_size() < EPM_SETTINGS['MIN_POOL_SIZE']

def run_mvl_cycle():
    """
    Run a single MVL (main value loop) cycle: generate/test DNA, update pool, and log results.
    Adds error handling for external calls and calculations.
    Logs DNA_GENERATED_MVL event after each cycle.
    """
    global cycle_counter, mutation_strength
    logger = logging.getLogger(__name__)
    cycle_counter += 1
    logger.info(f"=== MVL Cycle {cycle_counter} ===")
    if should_generate_new_dna():
        logger.info("Trigger: Generating new candidate DNA.")
        seed = LogicDNA.seed_rsi_buy()
        candidate = mutate_dna(seed, mutation_strength)
        try:
            cpu_load = get_cpu_load()
        except Exception as e:
            logger.warning(f"LEE: Failed to get CPU load: {e}. Using safe default (99.0)")
            cpu_load = 99.0
        try:
            market_data = get_dummy_market_data(LEE_SETTINGS['DUMMY_MARKET_DATA_SIZE'])
            test_results = run_nanostrat_test(candidate, market_data)
        except Exception as e:
            logger.warning(f"LEE: Error during nanostrat test: {e}. Using default test results.")
            test_results = {'virtual_pnl': 0.0, 'other_metrics': {}}
        try:
            fitness = evaluate_dna_fitness(test_results, cpu_load)
        except Exception as e:
            logger.warning(f"LEE: Error during fitness evaluation: {e}. Using 'discard_performance'.")
            fitness = 'discard_performance'
        logger.info(f"MVL Outcome: DNA={candidate} | TestResults={test_results} | CPU={cpu_load:.2f}% | Fitness={fitness}")
        # Structured logging for DNA_GENERATED_MVL
        log_event(
            'DNA_GENERATED_MVL',
            {
                'dna_id': candidate.id,
                'parent_id': candidate.parent_id,
                'initial_parameters': candidate.to_dict(),
                'mvl_test_pnl': test_results.get('virtual_pnl', None),
                'mvl_test_activations': test_results.get('activations', None),
                'mvl_cpu_load': cpu_load,
                'mvl_outcome': fitness,
            }
        )
        # Optional: MVP feedback tweak
        if fitness == 'discard_performance':
            mutation_strength = max(0.05, mutation_strength * 0.95)
        elif fitness == 'survived_mvl':
            mutation_strength = min(0.2, mutation_strength * 1.05)
            epm.add_dna_to_pool(candidate)
            # Structured logging for EPM_DNA_ADDED
            log_event(
                'EPM_DNA_ADDED',
                {
                    'dna_id': candidate.id,
                    'initial_fitness_from_mvl': fitness,
                }
            )
    else:
        logger.info("No new DNA generated this cycle.")

# --- RUN_CONFIGURATION_SNAPSHOT ---
def get_hardware_status():
    return {
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_load_percent': psutil.cpu_percent(interval=0.1),
        'virtual_memory': dict(psutil.virtual_memory()._asdict()),
    }

def check_log_file_accessible(log_file):
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('')
        return True
    except Exception as e:
        return str(e)

# Compose config snapshot
def log_run_configuration_snapshot():
    log_file = GENERAL_SETTINGS['STRUCTURED_DATA_LOG_FILE']
    log_event('RUN_CONFIGURATION_SNAPSHOT', {
        'timestamp_start_run': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
        'bot_version': '0.1.0-dev',
        'LEE_SETTINGS': dict(LEE_SETTINGS),
        'EPM_SETTINGS': dict(EPM_SETTINGS),
        'SO_SETTINGS': dict(SO_SETTINGS),
        'FERAL_CALIBRATOR_SETTINGS': dict(FERAL_CALIBRATOR_SETTINGS) if 'FERAL_CALIBRATOR_SETTINGS' in globals() else {},
        'LOGGING_SETTINGS': dict(LOGGING_SETTINGS),
        'GENERAL_SETTINGS': dict(GENERAL_SETTINGS),
        'META_PARAM_SETTINGS': {k: v for k, v in globals().items() if k.startswith('META_PARAM_') or k == 'ENABLE_META_SELF_TUNING'},
        'hardware_status': get_hardware_status(),
        'log_file_accessible': check_log_file_accessible(log_file),
        'datalogger_initialized': isinstance(logger_instance, object),
    })

# --- RUN_COMPLETED_SUMMARY ---
def log_run_completed_summary(start_time, dnas_generated, candidates_graduated, ab_tests, meta_param_events, final_pnl, critical_errors):
    end_time = time.time()
    log_event('RUN_COMPLETED_SUMMARY', {
        'timestamp_end_run': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(end_time)),
        'run_duration_sec': end_time - start_time,
        'total_dnas_generated': dnas_generated,
        'total_candidates_graduated': candidates_graduated,
        'total_ab_tests': ab_tests,
        'total_meta_param_self_adjusted': meta_param_events,
        'final_portfolio_pnl': final_pnl,
        'critical_errors_encountered': critical_errors,
    })

# --- Signal handler for graceful shutdown ---
shutdown_requested = False
def handle_sigint(sig, frame):
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
                 complexity_weights: dict):
        self.population_size = population_size
        self.mutation_rate_parametric = mutation_rate_parametric
        self.mutation_rate_structural = mutation_rate_structural
        self.crossover_rate = crossover_rate
        self.elitism_percentage = elitism_percentage
        self.random_injection_percentage = random_injection_percentage
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.complexity_weights = complexity_weights.copy() if complexity_weights else {}
        self.population: List[LogicDNA_v1] = []
        self.generation_counter = 0  # Track generation number
        # Motif seeding influence factor (probability to seed a motif in random tree)
        self.mle_motif_seeding_influence_factor = 0.15
        # Define available indicators and their valid lookback periods
        self.AVAILABLE_INDICATORS = {
            'SMA': [10, 20, 50],
            'EMA': [10, 20, 50],
            'RSI': [14],
            'VOLATILITY': [20]
        }

    def initialize_population(self, seed_dna_templates: Optional[List[LogicDNA_v1]] = None):
        """
        Initialize the population using hybrid seeding (some from templates, some random).
        Set generation_born=0 for all new individuals.
        """
        self.population = []
        n_seed = int(0.2 * self.population_size) if seed_dna_templates else 0
        n_random = self.population_size - n_seed
        # Seeded individuals
        if seed_dna_templates:
            for _ in range(n_seed):
                template = random.choice(seed_dna_templates)
                # Copy and (optionally) mutate slightly (for now, just copy)
                dna = template.copy()
                dna.generation_born = 0
                self.population.append(dna)
        # Random individuals
        for _ in range(n_random):
            tree = self._create_random_valid_tree(generation_born=0)
            self.population.append(tree)

    def _create_random_valid_tree(self, generation_born=None, mle_bias=None) -> LogicDNA_v1:
        """
        Generate a random valid LogicDNA_v1 tree respecting max_depth, max_nodes, and node type rules.
        If mle_bias/seed_motifs is present, with probability self.mle_motif_seeding_influence_factor, seed a motif.
        """
        # Motif seeding logic
        motifs = []
        if mle_bias and 'seed_motifs' in mle_bias and mle_bias['seed_motifs']:
            motifs = sorted(mle_bias['seed_motifs'].items(), key=lambda x: -x[1])
        if motifs and random.random() < self.mle_motif_seeding_influence_factor:
            motif, _ = random.choice(motifs[:min(5, len(motifs))])  # Pick from top 5
            # Simple: If motif is Indicator_X_Y_Used, create a ConditionNode with that indicator/period
            if motif.startswith('Indicator_') and '_Used' in motif:
                try:
                    _, ind, period, _ = motif.split('_')
                    period = int(period)
                    node = ConditionNode(
                        indicator_id=ind,
                        comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                        threshold_value=random.uniform(5, 100),
                        lookback_period_1=period,
                        lookback_period_2=None
                    )
                    tree = LogicDNA_v1(root_node=node, generation_born=generation_born)
                    if tree.is_valid(self.max_depth, self.max_nodes):
                        return tree
                except Exception as e:
                    pass  # Fallback to normal random
            # TODO: For more complex motifs (Condition-Action, Composite), parse and build tree segment
            # For now, fallback to normal random if not a simple indicator motif
        def _random_node(depth):
            if depth >= self.max_depth:
                # Must be an ActionNode at max depth
                return ActionNode(action_type=random.choice(["BUY", "SELL", "HOLD"]), size_factor=random.uniform(0.0, 1.0))
            node_type = random.choice(["Condition", "Action", "Composite", "Sequence"])
            if node_type == "Action":
                return ActionNode(action_type=random.choice(["BUY", "SELL", "HOLD"]), size_factor=random.uniform(0.0, 1.0))
            elif node_type == "Condition":
                indicator_id = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
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
                    lookback_period_2=None
                )
            elif node_type == "Composite":
                return CompositeNode(
                    logical_operator=random.choice(["AND", "OR"]),
                    child1=_random_node(depth+1),
                    child2=_random_node(depth+1)
                )
            elif node_type == "Sequence":
                return SequenceNode(
                    child1=_random_node(depth+1),
                    child2=_random_node(depth+1)
                )
            else:
                # Fallback to ActionNode
                return ActionNode(action_type="HOLD", size_factor=0.0)
        for _ in range(10):  # Try up to 10 times to get a valid tree
            root = _random_node(1)
            tree = LogicDNA_v1(root_node=root, generation_born=generation_born)
            if tree.is_valid(self.max_depth, self.max_nodes):
                return tree
        return LogicDNA_v1(root_node=ActionNode(action_type="HOLD", size_factor=0.0), generation_born=generation_born)

    def _run_backtest_for_dna(self, dna, historical_ohlcv: pd.DataFrame, precomputed_indicators: pd.DataFrame, backtest_config: dict) -> dict:
        """
        Backtest a LogicDNA_v1 individual on historical data.
        Simulates a simple portfolio with buy/sell/hold actions, applies transaction costs, and logs trades.
        Returns a dict of performance metrics.
        """
        initial_capital = backtest_config.get('initial_capital', 10000.0)
        transaction_cost_pct = backtest_config.get('transaction_cost_pct', 0.001)
        cash = initial_capital
        position = 0.0
        position_entry_price = 0.0
        equity_curve = []
        trade_log = []
        trade_count = 0
        last_action = 'HOLD'
        for i in range(len(historical_ohlcv)):
            row = historical_ohlcv.iloc[i]
            indicators_row = precomputed_indicators.iloc[i]
            available_indicators = indicators_row.to_dict()
            # --- Patch: Add VOLATILITY_20 alias if realized_vol_20 is present ---
            if 'realized_vol_20' in available_indicators:
                available_indicators['VOLATILITY_20'] = available_indicators['realized_vol_20']
            market_state = row.to_dict()
            ts = row['timestamp'] if 'timestamp' in row else i
            action_signal = dna.evaluate(market_state, available_indicators)
            price = row['close']
            if np.isnan(price) or any(np.isnan(list(available_indicators.values()))):
                equity_curve.append(cash + position * price if not np.isnan(price) else cash)
                continue
            action, size_factor = action_signal if isinstance(action_signal, tuple) and len(action_signal) == 2 else ("HOLD", 0.0)
            if action == 'BUY' and position == 0:
                max_affordable = cash / (price * (1 + transaction_cost_pct))
                size = max_affordable * size_factor
                cost = size * price * (1 + transaction_cost_pct)
                if cost <= cash and size > 0:
                    cash -= cost
                    position = size
                    position_entry_price = price
                    trade_ts = row['timestamp']
                    if hasattr(trade_ts, 'isoformat'):
                        trade_ts = trade_ts.isoformat()
                    else:
                        trade_ts = str(trade_ts)
                    trade_log.append({'timestamp': trade_ts, 'type': 'BUY', 'price': float(price), 'size': float(size)})
                    trade_count += 1
                    last_action = 'BUY'
            elif action == 'SELL' and position > 0:
                proceeds = position * price * (1 - transaction_cost_pct)
                cash += proceeds
                trade_ts = row['timestamp']
                if hasattr(trade_ts, 'isoformat'):
                    trade_ts = trade_ts.isoformat()
                else:
                    trade_ts = str(trade_ts)
                trade_log.append({'timestamp': trade_ts, 'type': 'SELL', 'price': float(price), 'size': float(position)})
                position = 0.0
                position_entry_price = 0.0
                trade_count += 1
                last_action = 'SELL'
            equity = cash + position * price
            equity_curve.append(equity)
        # Final equity
        final_equity = cash + position * historical_ohlcv.iloc[-1]['close']
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        profit = final_equity - initial_capital
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 2 else 0.0
        max_drawdown = 0.0
        peak = equity_curve[0] if equity_curve else initial_capital
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        # Ensure all metrics are JSON serializable
        metrics = {
            'metric_profit': float(profit),
            'metric_sharpe_ratio': float(sharpe),
            'metric_max_drawdown': float(max_drawdown),
            'metric_trade_count': int(trade_count),
            'metric_final_equity': float(final_equity),
            'metric_initial_capital': float(initial_capital),
            'trade_log': trade_log,
        }
        return metrics

    def _evaluate_population(self, active_persona: 'MarketPersona', market_data_for_evaluation: dict):
        """
        Evaluate each LogicDNA_v1 in the population using the real backtester.
        Expects market_data_for_evaluation to be a dict with keys:
            'ohlcv': pd.DataFrame
            'indicators': pd.DataFrame
            'backtest_config': dict
            'active_persona_name': str
            'ces_vector': dict
            'performance_log_path': str
            'current_generation': int
        Logs performance for MLE after each evaluation.
        """
        ohlcv = market_data_for_evaluation['ohlcv']
        indicators = market_data_for_evaluation['indicators']
        backtest_config = market_data_for_evaluation['backtest_config']
        active_persona_name = market_data_for_evaluation.get('active_persona_name', 'UNKNOWN')
        ces_vector = market_data_for_evaluation.get('ces_vector', {})
        performance_log_path = market_data_for_evaluation.get('performance_log_path', 'performance_log.csv')
        current_generation = market_data_for_evaluation.get('current_generation', getattr(self, 'generation_counter', 0))
        # Prepare log file
        log_exists = os.path.exists(performance_log_path)
        with open(performance_log_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'dna_id', 'generation_born', 'current_generation_evaluated',
                'logic_dna_structure_representation', 'performance_metrics',
                'fitness_score', 'active_persona_name', 'ces_vector_at_evaluation_time', 'timestamp_of_evaluation'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not log_exists:
                writer.writeheader()
            for dna in self.population:
                performance_metrics = self._run_backtest_for_dna(dna, ohlcv, indicators, backtest_config)
                node_count, tree_depth = dna.calculate_complexity()
                complexity_score = (
                    self.complexity_weights.get('nodes', 1.0) * node_count +
                    self.complexity_weights.get('depth', 1.0) * tree_depth
                )
                fitness = active_persona.calculate_fitness(performance_metrics, complexity_score)
                dna.fitness = fitness
                dna.performance_metrics = performance_metrics  # Optionally store for logging
                # --- Fix: Always log the string representation ---
                if hasattr(dna, 'to_string_representation') and callable(getattr(dna, 'to_string_representation')):
                    structure_repr = dna.to_string_representation()
                else:
                    structure_repr = ''
                writer.writerow({
                    'dna_id': getattr(dna, 'dna_id', ''),
                    'generation_born': getattr(dna, 'generation_born', ''),
                    'current_generation_evaluated': current_generation,
                    'logic_dna_structure_representation': structure_repr,
                    'performance_metrics': json.dumps(performance_metrics),
                    'fitness_score': fitness,
                    'active_persona_name': active_persona_name,
                    'ces_vector_at_evaluation_time': json.dumps(ces_vector),
                    'timestamp_of_evaluation': datetime.now().isoformat()
                })

    def _select_parents(self):
        """
        Select elites and parents for reproduction using elitism and tournament selection.
        Returns (elites, parents_for_reproduction)
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(self.population, key=lambda d: getattr(d, 'fitness', 0.0), reverse=True)
        n_elite = max(1, int(self.elitism_percentage * self.population_size))
        elites = [dna.copy() for dna in sorted_pop[:n_elite]]
        # Tournament selection for parents
        parents = []
        tournament_size = 3
        while len(parents) < self.population_size:
            candidates = random.sample(sorted_pop, min(tournament_size, len(sorted_pop)))
            winner = max(candidates, key=lambda d: getattr(d, 'fitness', 0.0))
            parents.append(winner.copy())
        return elites, parents

    def run_evolutionary_cycle(self, active_persona: 'MarketPersona', market_data_for_evaluation: dict, mle_bias: dict = None, ces_info: dict = None):
        """
        Run one full evolutionary cycle: evaluate, select, reproduce, inject randoms, update population.
        Accepts optional mle_bias and ces_info for feedback integration.
        """
        # Adjust mutation rates based on CES stress score (if provided)
        orig_mutation_rate_structural = self.mutation_rate_structural
        orig_mutation_rate_parametric = self.mutation_rate_parametric
        if ces_info is not None and isinstance(ces_info, dict):
            # Use volatility, liquidity, trend for composite stress
            v = ces_info.get('volatility', 0.5)
            l = ces_info.get('liquidity', 0.5)
            t = ces_info.get('trend', 0.5)
            # Composite stress: higher if volatility is high, liquidity is low, trend is high
            composite_stress = (v + (1-l) + t) / 3.0
            # Adjust rates: high stress -> increase, low stress -> decrease
            adj = 0.05 if composite_stress > 0.7 else (-0.03 if composite_stress < 0.4 else 0.0)
            self.mutation_rate_structural = min(1.0, max(0.01, orig_mutation_rate_structural + adj))
            self.mutation_rate_parametric = min(1.0, max(0.01, orig_mutation_rate_parametric + adj))
            print(f"[LEE] Adjusted mutation rates for CES: structural={self.mutation_rate_structural:.3f}, parametric={self.mutation_rate_parametric:.3f} (composite_stress={composite_stress:.2f})")
        # Print MLE bias influence (placeholder)
        if mle_bias is not None:
            print(f"[LEE] MLE Bias (seed_motifs): {mle_bias.get('seed_motifs', {})}")
            print(f"[LEE] MLE Bias (operator_biases): {mle_bias.get('recommended_operator_biases', {})}")
        # Print CES info influence (placeholder)
        if ces_info is not None:
            print(f"[LEE] CES Info: {ces_info}")
        self._evaluate_population(active_persona, market_data_for_evaluation)
        self.generation_counter += 1
        fitnesses = [getattr(dna, 'fitness', 0.0) for dna in self.population]
        best = max(fitnesses) if fitnesses else None
        avg = sum(fitnesses)/len(fitnesses) if fitnesses else None
        print(f"Generation: {self.generation_counter} - Best Fitness: {best:.3f}, Avg Fitness: {avg:.3f}")
        elites, parents = self._select_parents()
        num_random_injections = int(self.random_injection_percentage * self.population_size)
        num_offspring_needed = self.population_size - len(elites) - num_random_injections
        offspring = self._reproduce(parents, num_offspring_needed, mle_bias)
        random_new = [self._create_random_valid_tree(mle_bias=mle_bias) for _ in range(num_random_injections)]
        self.population = elites + offspring + random_new
        # --- Diversity metric ---
        unique_structures = set()
        for dna in self.population:
            if hasattr(dna, 'to_string_representation'):
                unique_structures.add(dna.to_string_representation())
        diversity = len(unique_structures)
        diversity_pct = diversity / self.population_size
        print(f"[LEE] Population diversity: {diversity} unique structures ({diversity_pct:.2%})")
        # Log to summary log
        log_event('POP_DIVERSITY', {'generation': self.generation_counter, 'diversity': diversity, 'diversity_pct': diversity_pct})
        # Diversity collapse warning
        if not hasattr(self, '_diversity_low_counter'):
            self._diversity_low_counter = 0
        if diversity_pct < 0.2:
            self._diversity_low_counter += 1
            if self._diversity_low_counter >= 3:
                print("WARNING: Population Diversity Low!")
                log_event('DIVERSITY_COLLAPSE_WARNING', {'generation': self.generation_counter, 'diversity': diversity, 'diversity_pct': diversity_pct})
        else:
            self._diversity_low_counter = 0
        # Restore original mutation rates
        self.mutation_rate_structural = orig_mutation_rate_structural
        self.mutation_rate_parametric = orig_mutation_rate_parametric
        logging.info(f"LEE Generation: Best fitness={best}, Avg fitness={avg}")

    def _reproduce(self, parents_for_reproduction, num_offspring_needed, mle_bias=None):
        """
        Generate offspring using crossover and mutation.
        Accepts optional mle_bias for motif/operator biasing (placeholder logic).
        """
        offspring = []
        current_generation = getattr(self, 'generation_counter', 0)
        motif_adopted = 0
        for _ in range(num_offspring_needed):
            used_motif = False  # Ensure always initialized
            # Motif bias logic is now in _structural_mutation/_create_random_valid_tree
            if random.random() < self.crossover_rate and len(parents_for_reproduction) > 1:
                p1, p2 = random.sample(parents_for_reproduction, 2)
                child1, child2 = self._crossover_trees(p1, p2)
                candidate = child1 if random.random() < 0.5 else child2
            else:
                candidate = random.choice(parents_for_reproduction).copy()
            # Structural mutation
            if random.random() < self.mutation_rate_structural:
                before = candidate.to_string_representation() if hasattr(candidate, 'to_string_representation') else ''
                candidate = self._structural_mutation(candidate, mle_bias=mle_bias)
                after = candidate.to_string_representation() if hasattr(candidate, 'to_string_representation') else ''
                if mle_bias and 'seed_motifs' in mle_bias and before != after:
                    # Heuristic: if structure changed and motifs present, count as adopted
                    used_motif = True
            # Parametric mutation (always)
            candidate = self._parametric_mutation(candidate)
            if candidate.is_valid(self.max_depth, self.max_nodes):
                candidate.generation_born = current_generation
                offspring.append(candidate)
            else:
                tree = self._create_random_valid_tree(generation_born=current_generation, mle_bias=mle_bias)
                offspring.append(tree)
            if used_motif:
                motif_adopted += 1
        print(f"[LEE] Motif adoption rate this generation: {motif_adopted}/{num_offspring_needed}")
        from src.data_logger import log_event
        log_event('MOTIF_ADOPTION', {'generation': getattr(self, 'generation_counter', 0), 'motif_adopted': motif_adopted, 'num_offspring': num_offspring_needed})
        return offspring

    def _parametric_mutation(self, dna: LogicDNA_v1) -> LogicDNA_v1:
        """
        Mutate numerical parameters of nodes in a copy of the DNA.
        For ConditionNodes, only mutate to valid indicator/period combinations.
        """
        dna_copy = dna.copy()
        def _mutate(node):
            if isinstance(node, ConditionNode):
                # Mutate indicator_id and lookback_period_1 only to valid combos
                if random.random() < 0.2:
                    indicator_id = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
                    node.indicator_id = indicator_id
                    if indicator_id == 'RSI':
                        node.lookback_period_1 = 14
                    elif indicator_id == 'VOLATILITY':
                        node.lookback_period_1 = 20
                    else:
                        node.lookback_period_1 = random.choice(self.AVAILABLE_INDICATORS[indicator_id])
                else:
                    # Only mutate period if not changing indicator
                    if random.random() < 0.5:
                        if node.indicator_id == 'RSI':
                            node.lookback_period_1 = 14
                        elif node.indicator_id == 'VOLATILITY':
                            node.lookback_period_1 = 20
                        else:
                            node.lookback_period_1 = random.choice(self.AVAILABLE_INDICATORS[node.indicator_id])
                if random.random() < 0.5:
                    node.threshold_value += random.uniform(-2, 2)
            elif isinstance(node, ActionNode):
                if random.random() < 0.5:
                    node.size_factor = min(1.0, max(0.0, node.size_factor + random.uniform(-0.1, 0.1)))
            elif isinstance(node, (CompositeNode, SequenceNode)):
                _mutate(node.child1)
                _mutate(node.child2)
        _mutate(dna_copy.root_node)
        return dna_copy

    def _structural_mutation(self, dna: LogicDNA_v1, mle_bias: dict = None) -> LogicDNA_v1:
        """
        Randomly add, prune, or replace a node/subtree in a copy of the DNA.
        If mle_bias/seed_motifs is present, with probability, inject a motif node/branch.
        """
        dna_copy = dna.copy()
        motif_inject_prob = 0.15
        if mle_bias and 'recommended_operator_biases' in mle_bias:
            motif_inject_prob = mle_bias['recommended_operator_biases'].get('structural_mutation_rate_adjustment_factor', 0.15)
        motifs = []
        if mle_bias and 'seed_motifs' in mle_bias and mle_bias['seed_motifs']:
            motifs = sorted(mle_bias['seed_motifs'].items(), key=lambda x: -x[1])
        if motifs and random.random() < motif_inject_prob:
            motif, _ = random.choice(motifs[:min(5, len(motifs))])
            if motif.startswith('Indicator_') and '_Used' in motif:
                try:
                    _, ind, period, _ = motif.split('_')
                    period = int(period)
                    new_node = ConditionNode(
                        indicator_id=ind,
                        comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                        threshold_value=random.uniform(5, 100),
                        lookback_period_1=period,
                        lookback_period_2=None
                    )
                    dna_copy.add_node_at_random_valid_point(new_node)
                    return dna_copy
                except Exception as e:
                    pass  # Fallback to normal mutation
            # TODO: For more complex motifs, parse and inject as subtree
        # Fallback to normal mutation
        mutation_type = random.choice(['add', 'prune', 'replace'])
        if mutation_type == 'add':
            indicator_id = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
            if indicator_id == 'RSI':
                lookback_period_1 = 14
            elif indicator_id == 'VOLATILITY':
                lookback_period_1 = 20
            else:
                lookback_period_1 = random.choice(self.AVAILABLE_INDICATORS[indicator_id])
            new_node = ConditionNode(
                indicator_id=indicator_id,
                comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                threshold_value=random.uniform(5, 100),
                lookback_period_1=lookback_period_1,
                lookback_period_2=None
            )
            dna_copy.add_node_at_random_valid_point(new_node)
        elif mutation_type == 'prune':
            dna_copy.prune_random_subtree()
        elif mutation_type == 'replace':
            node = dna_copy.get_random_node()
            if node and hasattr(node, 'node_id'):
                if random.random() < 0.5:
                    dna_copy.replace_node(node.node_id, ActionNode(action_type="HOLD", size_factor=0.0))
                else:
                    indicator_id = random.choice(list(self.AVAILABLE_INDICATORS.keys()))
                    if indicator_id == 'RSI':
                        lookback_period_1 = 14
                    elif indicator_id == 'VOLATILITY':
                        lookback_period_1 = 20
                    else:
                        lookback_period_1 = random.choice(self.AVAILABLE_INDICATORS[indicator_id])
                    new_cond = ConditionNode(
                        indicator_id=indicator_id,
                        comparison_operator=random.choice(["GREATER_THAN", "LESS_THAN", "EQUAL_TO"]),
                        threshold_value=random.uniform(5, 100),
                        lookback_period_1=lookback_period_1,
                        lookback_period_2=None
                    )
                    dna_copy.replace_node(node.node_id, new_cond)
        return dna_copy

    def _crossover_trees(self, parent1: LogicDNA_v1, parent2: LogicDNA_v1):
        """
        Single-point subtree exchange between two parents.
        """
        p1 = parent1.copy()
        p2 = parent2.copy()
        node1 = p1.get_random_subtree_point()
        node2 = p2.get_random_subtree_point()
        if node1 and node2 and hasattr(node1, 'node_id') and hasattr(node2, 'node_id'):
            # Swap subtrees
            p1.replace_node(node1.node_id, node2.copy())
            p2.replace_node(node2.node_id, node1.copy())
        # Ensure validity
        if not p1.is_valid(self.max_depth, self.max_nodes):
            p1 = self._create_random_valid_tree()
        if not p2.is_valid(self.max_depth, self.max_nodes):
            p2 = self._create_random_valid_tree()
        return p1, p2 