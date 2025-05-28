"""
System Orchestrator for Project Crypto Bot Peter.

This module defines the `SystemOrchestrator` class, which acts as the central
coordinator for the trading bot. It is responsible for:
- Loading and validating system configurations.
- Initializing and managing other core modules such as the Logic Evolution
  Engine (LEE), Meta-Learning Engine (MLE), and Contextual Environment
  Scorer (CES).
- Managing the lifecycle of trading strategy candidates, including promotion,
  demotion, and A/B testing.
- Orchestrating the overall system flow, including evolutionary cycles and
  the application of market personas.
- Handling RNG state for reproducibility.
"""
"""
System Orchestrator for Project Crypto Bot Peter.

This module defines the `SystemOrchestrator` class, which acts as the central
coordinator for the trading bot. It is responsible for:
- Loading and validating system configurations.
- Initializing and managing other core modules such as the Logic Evolution
  Engine (LEE), Meta-Learning Engine (MLE), and Contextual Environment
  Scorer (CES).
- Managing the lifecycle of trading strategy candidates, including promotion,
  demotion, and A/B testing.
- Orchestrating the overall system flow, including evolutionary cycles and
  the application of market personas.
- Handling RNG state for reproducibility.
"""
import logging
import numpy as np
import math
from src.logic_dna import mutate_dna, LogicDNA
from src.nanostrat import run_nanostrat_test
from collections import deque, defaultdict
from settings import SO_SETTINGS
import psutil
from src.data_logger import log_event
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Deque, TYPE_CHECKING
import random
import pickle
import os
import numpy as np # Added
import atexit # Added

from src.meta_parameter_monitor import MetaParameterMonitor
import settings  # <-- Add this import
import json
# from src.lee import LEE # Moved to TYPE_CHECKING
from src.market_persona import MarketPersona
from src.logic_dna import LogicDNA_v1
from src.mle_engine import MLE_v0_1
from src.ces_module import CES_v1_0
from src.lee import LEE # Moved out of TYPE_CHECKING


if TYPE_CHECKING:
    # from src.lee import LEE # Moved out
    from src.experimental_pool import PoolDNAEntry # Assuming this is the type of graduated_entry

logger = logging.getLogger(__name__)

# Constants with type hints
MAX_CANDIDATE_SLOTS: int = SO_SETTINGS.get('MAX_CANDIDATE_SLOTS', 2) # Example if from settings
CANDIDATE_VIRTUAL_CAPITAL: float = SO_SETTINGS.get('CANDIDATE_VIRTUAL_CAPITAL', 10.0)
CANDIDATE_MIN_SHARPE: float = SO_SETTINGS.get('CANDIDATE_MIN_SHARPE', 0.3)
CANDIDATE_MIN_PNL: float = SO_SETTINGS.get('CANDIDATE_MIN_PNL', 5.0)
CANDIDATE_MIN_TICKS: int = SO_SETTINGS.get('CANDIDATE_MIN_TICKS', 100)
CANDIDATE_MIN_ACTIVATIONS: int = SO_SETTINGS.get('CANDIDATE_MIN_ACTIVATIONS', 3)
CANDIDATE_EPSILON: float = SO_SETTINGS.get('CANDIDATE_EPSILON', 1e-6)
CANDIDATE_MAX_LIFE: int = SO_SETTINGS.get('CANDIDATE_MAX_LIFE', 2000)
CANDIDATE_SHARPE_DROP_TRIGGER: float = SO_SETTINGS.get('CANDIDATE_SHARPE_DROP_TRIGGER', 0.3)
AB_TEST_TICKS: int = SO_SETTINGS.get('AB_TEST_TICKS', 200)
FITNESS_WEIGHTS: Dict[str, float] = SO_SETTINGS.get('FITNESS_WEIGHTS', {'sharpe': 0.4, 'regime_consistency': 0.3, 'diversification': 0.3})
AB_SHARPE_THRESHOLD: float = SO_SETTINGS.get('AB_SHARPE_THRESHOLD', 0.10)
AB_PNL_THRESHOLD: float = SO_SETTINGS.get('AB_PNL_THRESHOLD', 0.05)
INFLUENCE_MULTIPLIER_MAX: float = SO_SETTINGS.get('INFLUENCE_MULTIPLIER_MAX', 1.5)
INFLUENCE_MULTIPLIER_MIN: float = SO_SETTINGS.get('INFLUENCE_MULTIPLIER_MIN', 0.5)
INFLUENCE_MULTIPLIER_STEP_UP: float = SO_SETTINGS.get('INFLUENCE_MULTIPLIER_STEP_UP', 0.1)
INfluence_MULTIPLIER_STEP_DOWN: float = SO_SETTINGS.get('INFLUENCE_MULTIPLIER_STEP_DOWN', 0.1) # Typo in original? Assuming INFLUENCE
INFLUENCE_SCORE_HIGH_PCT: float = SO_SETTINGS.get('INFLUENCE_SCORE_HIGH_PCT', 0.75)
INFLUENCE_SCORE_LOW_PCT: float = SO_SETTINGS.get('INFLUENCE_SCORE_LOW_PCT', 0.25)
COOLDOWN_TICKS_AFTER_TWEAK: int = SO_SETTINGS.get('COOLDOWN_TICKS_AFTER_TWEAK', 750)


def check_settings_dict(settings_dict: Dict[str, Any], required_keys: List[str], dict_name: str) -> None:
    missing: List[str] = [k for k in required_keys if k not in settings_dict]
    if missing:
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name}) # type: ignore
        raise RuntimeError(f"CRITICAL: Missing keys in {dict_name}: {missing}")

class ConflictResolver:
    """
    MVP stub for conflict detection and resolution between system influences.
    For now, always returns False for conflicts and passes through parameters.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config

    def detect_conflicts(self, current_params: Dict[str, Any], influences: Dict[str, Any]) -> bool:
        """
        Detects if there are conflicting influences on key parameters.
        For MVP, always returns False.
        """
        logger.debug("ConflictResolver.detect_conflicts called (MVP: always returns False)")
        return False

    def resolve(self, current_params: Dict[str, Any], influences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves conflicts by applying a simple rule (prioritize safety, etc.).
        For MVP, just returns current_params unchanged.
        """
        logger.info(f"ConflictResolver.resolve called (MVP: returning current_params) for influences: {influences}")
        return current_params

class RegimeClassifier:
    """
    Classifies market regime using rolling window of price data.
    Regimes: HIGH_VOL, LOW_VOL, UPTREND_STRONG, DOWNTREND_STRONG, SIDEWAYS_CHOPPY
    """
    def __init__(self, window: int = 50):
        self.window: int = window
        self.price_buffer: Deque[float] = deque(maxlen=window)
        self.last_regime: Optional[Dict[str, str]] = None

    def update(self, price: float) -> Optional[Dict[str, str]]:
        self.price_buffer.append(price)
        return self.classify()

    def classify(self) -> Optional[Dict[str, str]]:
        if len(self.price_buffer) < 10: # Need enough data points
            return None
        
        prices: np.ndarray = np.array(self.price_buffer, dtype=float)
        returns: np.ndarray = np.diff(prices)
        if len(returns) == 0: return None # Not enough data for returns

        vol: float = np.std(returns)
        
        sma_window: int = min(20, len(prices)) # Ensure SMA window is not larger than available prices
        if sma_window < 2: return None # Not enough data for SMA trend calculation

        sma: np.ndarray = np.convolve(prices, np.ones(sma_window)/sma_window, mode='valid')
        if len(sma) < 2: return None # Not enough data for trend calculation from SMA

        trend: float = (sma[-1] - sma[0]) / sma_window if sma_window > 0 else 0.0
        
        vol_regime: str
        if vol > 1.0: vol_regime = 'HIGH_VOL'
        elif vol < 0.3: vol_regime = 'LOW_VOL'
        else: vol_regime = 'MEDIUM_VOL'
        
        trend_regime: str
        if trend > 0.05: trend_regime = 'UPTREND_STRONG'
        elif trend < -0.05: trend_regime = 'DOWNTREND_STRONG'
        else: trend_regime = 'SIDEWAYS_CHOPPY'
            
        current_regime: Dict[str, str] = {'VOLATILITY': vol_regime, 'TREND': trend_regime}
        
        if current_regime != self.last_regime:
            log_event( # type: ignore
                'REGIME_CHANGE_DETECTED', # type: ignore
                { # type: ignore
                    'old_regime': self.last_regime, # type: ignore
                    'new_regime': current_regime, # type: ignore
                    'timestamp': datetime.now().isoformat(), # type: ignore
                } # type: ignore
            ) # type: ignore
            logger.info(f"SO RegimeClassifier: Regime changed to {current_regime}")
            self.last_regime = current_regime
        return current_regime

class CandidateStrategySlot:
    """
    Represents a candidate strategy slot in the SO, tracking a DNA's performance and lifecycle.
    Args:
        pool_entry (PoolDNAEntry): The graduated pool entry containing the DNA.
    Attributes:
        dna (LogicDNA): The DNA instance (with unique id).
        ... (other attributes as before)
    """
    def __init__(self, pool_entry: 'PoolDNAEntry'): # Use forward reference for PoolDNAEntry
        self.dna: LogicDNA = pool_entry.dna # type: ignore # Assuming PoolDNAEntry has a .dna attribute of type LogicDNA
        self.dedicated_virtual_capital_candidate: float = CANDIDATE_VIRTUAL_CAPITAL
        self.pnl_candidate: float = 0.0
        self.sharpe_candidate: float = 0.0
        self.ticks_in_candidate: int = 0
        self.activation_count_candidate: int = 0
        self.pnl_history: List[float] = []
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.ready_for_demotion: bool = False
        self.recent_high_sharpe: float = 0.0
        self.self_calibration_needed: bool = False
        self.influence_multiplier: float = 1.0
        self.last_tweak_implemented_at: int = -COOLDOWN_TICKS_AFTER_TWEAK

    def evaluate_tick(self, market_tick: Dict[str, Any], global_tick: Optional[int] = None) -> None:
        """
        Evaluate this candidate's DNA for the current market tick, updating PnL, Sharpe, and calibration triggers.
        """
        self.ticks_in_candidate += 1
        indicator_value: Optional[float] = market_tick.get(self.dna.trigger_indicator) # type: ignore
        
        if indicator_value is None:
            return # Cannot evaluate if indicator is missing

        op: str = self.dna.trigger_operator # type: ignore
        threshold: float = self.dna.trigger_threshold # type: ignore
        
        triggered: bool = (op == '<' and indicator_value < threshold) or \
                          (op == '>' and indicator_value > threshold)
        
        if triggered:
            self.activation_count_candidate += 1
            price_now: float = market_tick.get('price', 0.0)
            price_prev: float = market_tick.get('price_prev', price_now) # Use current if prev is missing
            
            pnl_this_tick: float = 0.0
            if self.dna.action_target.startswith('buy'): # type: ignore
                pnl_this_tick = self.dedicated_virtual_capital_candidate * (1.0 if price_now > price_prev else -1.0)
            elif self.dna.action_target.startswith('sell'): # type: ignore
                pnl_this_tick = self.dedicated_virtual_capital_candidate * (1.0 if price_now < price_prev else -1.0)
            
            self.pnl_candidate += pnl_this_tick
            self.pnl_history.append(pnl_this_tick)
            self.logger.info(f"CandidateSlot: DNA {self.dna.id} triggered | PnL: {pnl_this_tick} | TotalPnL: {self.pnl_candidate}")

        if self.pnl_history:
            mean_pnl: float = np.mean(self.pnl_history) # type: ignore
            std_dev_pnl: float = np.std(self.pnl_history) + CANDIDATE_EPSILON # type: ignore
            self.sharpe_candidate = mean_pnl / std_dev_pnl if std_dev_pnl > 0 else 0.0 # Avoid division by zero

            if self.sharpe_candidate > self.recent_high_sharpe:
                self.recent_high_sharpe = self.sharpe_candidate
            
            # Self-calibration trigger
            sharpe_drop_pct: float = (self.recent_high_sharpe - self.sharpe_candidate) / self.recent_high_sharpe if self.recent_high_sharpe > 0 else 0
            cooldown_passed: bool = global_tick is None or (global_tick - self.last_tweak_implemented_at > COOLDOWN_TICKS_AFTER_TWEAK)

            if sharpe_drop_pct > CANDIDATE_SHARPE_DROP_TRIGGER and cooldown_passed:
                self.self_calibration_needed = True
                self.logger.info(f"CandidateSlot: DNA {self.dna.id} triggered self-calibration | Sharpe dropped from {self.recent_high_sharpe:.2f} to {self.sharpe_candidate:.2f}")
        
        # Demotion check
        if self.ticks_in_candidate > CANDIDATE_MAX_LIFE and \
           (self.sharpe_candidate < CANDIDATE_MIN_SHARPE or self.pnl_candidate < CANDIDATE_MIN_PNL):
            self.ready_for_demotion = True
            self.logger.info(f"CandidateSlot: DNA {self.dna.id} flagged for demotion | Sharpe={self.sharpe_candidate:.2f} | PnL={self.pnl_candidate:.2f}")

    def is_ready_for_graduation(self) -> bool:
        """
        Check if this candidate is ready for graduation based on performance and activity.
        Returns:
            bool: True if ready for graduation.
        """
        return (self.ticks_in_candidate >= CANDIDATE_MIN_TICKS and
                self.activation_count_candidate >= CANDIDATE_MIN_ACTIVATIONS and
                self.pnl_candidate >= CANDIDATE_MIN_PNL and
                self.sharpe_candidate >= CANDIDATE_MIN_SHARPE)

    def run_self_calibration(self, recent_market_data: List[Dict[str, Any]], so_instance: 'SystemOrchestrator', regime: Optional[Dict[str, str]] = None) -> None:
        """
        Parameter Explorer & Tweak Assessor: Generate 2-3 tweaks, micro-sim each, compare to original, propose best to SO.
        Args:
            recent_market_data: List of dicts (market ticks) for micro-sim.
            so_instance: SystemOrchestrator instance for logging proposal.
            regime: Regime information for micro-sim.
        """
        self.logger.info(f"CandidateSlot: Running self-calibration for DNA {self.dna.id}") # type: ignore
        tweaks: List[LogicDNA] = [mutate_dna(self.dna, mutation_strength=0.1) for _ in range(3)] # type: ignore
        
        sim_results: List[Dict[str, Any]] = []
        sensitivity_profile_data: List[Dict[str, Any]] = []
        
        original_sim_result: Dict[str, Any] = run_nanostrat_test(self.dna, recent_market_data) # type: ignore
        sim_results.append({'dna': self.dna, 'result': original_sim_result, 'is_original': True})
        
        for tweaked_dna in tweaks:
            tweaked_sim_result: Dict[str, Any] = run_nanostrat_test(tweaked_dna, recent_market_data) # type: ignore
            sim_results.append({'dna': tweaked_dna, 'result': tweaked_sim_result, 'is_original': False})
            for param_name in ['trigger_threshold', 'action_value']:
                if getattr(tweaked_dna, param_name) != getattr(self.dna, param_name): # type: ignore
                    pnl: float = tweaked_sim_result.get('virtual_pnl', 0.0)
                    sharpe: float = pnl / (np.std([pnl]) + CANDIDATE_EPSILON) if pnl != 0 else 0.0 # Simplified Sharpe for single value
                    sensitivity_profile_data.append({
                        'param': param_name, 
                        'value': getattr(tweaked_dna, param_name), 
                        'pnl': pnl, 
                        'sharpe': sharpe
                    })

        best_sim: Dict[str, Any] = max(sim_results, key=lambda x: x['result'].get('virtual_pnl', float('-inf')))
        
        if not best_sim['is_original'] and best_sim['result'].get('virtual_pnl', 0.0) > original_sim_result.get('virtual_pnl', 0.0):
            best_dna_tweak: LogicDNA = best_sim['dna']
            pnl_uplift: float = best_sim['result'].get('virtual_pnl', 0.0) - original_sim_result.get('virtual_pnl', 0.0)
            self.logger.info(f"CandidateSlot: Best tweak found for DNA {self.dna.id}: {best_dna_tweak} | PnL uplift: {pnl_uplift:.2f}") # type: ignore
            
            regime_for_sim: Any = regime if regime else 'unknown'
            self.propose_tweak_to_so(best_dna_tweak, self.dna, best_sim['result'], original_sim_result, so_instance, regime_for_sim, sensitivity_profile_data) # type: ignore
        else:
            self.logger.info(f"CandidateSlot: No tweak outperformed original for DNA {self.dna.id}") # type: ignore
            
        self.self_calibration_needed = False

    def propose_tweak_to_so(self, 
                            tweak_dna: LogicDNA, 
                            orig_dna: LogicDNA, 
                            tweak_result: Dict[str, Any], 
                            orig_result: Dict[str, Any], 
                            so_instance: 'SystemOrchestrator', 
                            micro_sim_regimes: Optional[Any] = None, 
                            sensitivity_profile: Optional[List[Dict[str, Any]]] = None) -> None:
        so_instance.log_tweak_proposal(
            candidate_id=orig_dna.id, # type: ignore
            orig_params={
                'trigger_indicator': orig_dna.trigger_indicator, # type: ignore
                'trigger_operator': orig_dna.trigger_operator, # type: ignore
                'trigger_threshold': orig_dna.trigger_threshold, # type: ignore
                'context_regime_id': orig_dna.context_regime_id, # type: ignore
                'action_target': orig_dna.action_target, # type: ignore
                'action_type': orig_dna.action_type, # type: ignore
                'action_value': orig_dna.action_value, # type: ignore
                'resource_cost': getattr(orig_dna, 'resource_cost', 1) # type: ignore
            },
            tweak_params={
                'trigger_indicator': tweak_dna.trigger_indicator, # type: ignore
                'trigger_operator': tweak_dna.trigger_operator, # type: ignore
                'trigger_threshold': tweak_dna.trigger_threshold, # type: ignore
                'context_regime_id': tweak_dna.context_regime_id, # type: ignore
                'action_target': tweak_dna.action_target, # type: ignore
                'action_type': tweak_dna.action_type, # type: ignore
                'action_value': tweak_dna.action_value, # type: ignore
                'resource_cost': getattr(tweak_dna, 'resource_cost', 1) # type: ignore
            },
            orig_result=orig_result,
            tweak_result=tweak_result,
            micro_sim_regimes=micro_sim_regimes,
            sensitivity_profile=sensitivity_profile
        )

class ABTestManager:
    def __init__(self) -> None:
        self.active_tests: Dict[str, 'ABTestManager.ABTest'] = {} 

    class ABTest:
        def __init__(self, candidate_id: str, control_dna: LogicDNA, variant_dna: LogicDNA, start_tick: int, regime: Optional[Dict[str, str]]):
            self.candidate_id: str = candidate_id
            self.control_dna: LogicDNA = control_dna
            self.variant_dna: LogicDNA = variant_dna
            self.start_tick: int = start_tick
            self.regime: Optional[Dict[str, str]] = regime
            self.ticks: int = 0
            self.control_metrics: Dict[str, Any] = {'pnl': 0.0, 'activations': 0, 'pnl_history': []}
            self.variant_metrics: Dict[str, Any] = {'pnl': 0.0, 'activations': 0, 'pnl_history': []}
            self.finished: bool = False

        def step(self, market_tick: Dict[str, Any]) -> None:
            for _label, dna_instance, metrics_dict in [('control', self.control_dna, self.control_metrics), 
                                                       ('variant', self.variant_dna, self.variant_metrics)]:
                indicator_val: Optional[float] = market_tick.get(dna_instance.trigger_indicator) # type: ignore
                if indicator_val is None: continue

                operator: str = dna_instance.trigger_operator # type: ignore
                thresh: float = dna_instance.trigger_threshold # type: ignore
                is_triggered: bool = (operator == '<' and indicator_val < thresh) or \
                                   (operator == '>' and indicator_val > thresh)
                
                if is_triggered:
                    metrics_dict['activations'] += 1
                    current_price: float = market_tick.get('price', 0.0)
                    previous_price: float = market_tick.get('price_prev', current_price)
                    
                    tick_pnl: float = 0.0
                    if dna_instance.action_target.startswith('buy'): # type: ignore
                        tick_pnl = 1.0 if current_price > previous_price else -1.0
                    elif dna_instance.action_target.startswith('sell'): # type: ignore
                        tick_pnl = 1.0 if current_price < previous_price else -1.0
                    
                    metrics_dict['pnl'] += tick_pnl
                    metrics_dict['pnl_history'].append(tick_pnl)
            
            self.ticks += 1
            if self.ticks >= AB_TEST_TICKS:
                self.finished = True

        def summary(self) -> Dict[str, Any]:
            def calculate_sharpe(metrics_data: Dict[str, Any]) -> float:
                pnl_hist: List[float] = metrics_data.get('pnl_history', [])
                if not pnl_hist: return 0.0
                mean_ret: float = np.mean(pnl_hist) # type: ignore
                std_ret: float = np.std(pnl_hist) + CANDIDATE_EPSILON # type: ignore
                return mean_ret / std_ret if std_ret > 0 else 0.0

            return {
                'control': {
                    'pnl': self.control_metrics['pnl'],
                    'activations': self.control_metrics['activations'],
                    'sharpe': calculate_sharpe(self.control_metrics)
                },
                'variant': {
                    'pnl': self.variant_metrics['pnl'],
                    'activations': self.variant_metrics['activations'],
                    'sharpe': calculate_sharpe(self.variant_metrics)
                },
                'regime': self.regime,
                'ticks': self.ticks
            }

    def start_ab_test(self, candidate_id: str, control_dna: LogicDNA, variant_dna: LogicDNA, start_tick: int, regime: Optional[Dict[str, str]]) -> None:
        self.active_tests[candidate_id] = ABTestManager.ABTest(candidate_id, control_dna, variant_dna, start_tick, regime)

    def step_all(self, market_tick: Dict[str, Any]) -> List[str]:
        finished_tests_ids: List[str] = []
        for cid, ab_test_instance in self.active_tests.items():
            ab_test_instance.step(market_tick)
            if ab_test_instance.finished:
                finished_tests_ids.append(cid)
        return finished_tests_ids

    def get_finished(self) -> List[str]:
        return [cid for cid, ab_test_instance in self.active_tests.items() if ab_test_instance.finished]

    def pop_finished(self) -> Dict[str, Any]:
        finished_ids: List[str] = self.get_finished()
        results_dict: Dict[str, Any] = {cid: self.active_tests.pop(cid).summary() for cid in finished_ids}
        return results_dict

class SystemOrchestrator:
    """
    Minimal v1.0 SystemOrchestrator for Project Crypto Bot Peter.
    Loads configuration, manages LEE and MarketPersonas, and runs evolutionary cycles.

    RNG State Management:
        This feature allows for saving and loading the state of the pseudo-random
        number generator (RNG) used by the `random` module. This is crucial for
        ensuring the reproducibility of experiments or specific simulation runs.

        To enable this feature:
        1.  Set `rng_state_load_path` in the configuration file to the path of an
            existing RNG state file if you wish to start with a specific RNG state.
        2.  Set `rng_state_save_path` in the configuration file to the path where
            the RNG state should be saved at the end of a run.

        The `load_rng_state` method is called during initialization if
        `rng_state_load_path` is provided. The `save_rng_state` method is called
        at the end of `run_n_generations` if `rng_state_save_path` is provided.

        Security Note:
            RNG state files (often using pickle) should only be loaded from
            trusted sources, as loading a malicious pickle file can lead to
            arbitrary code execution.
        - Advanced Consideration: If Project Crypto Bot Peter is ever adapted
          for multi-process or heavily multi-threaded execution, the current
          global RNG state restoration mechanism for `random` and `numpy.random`
          may require careful review and potential modification (e.g., using
          per-process/per-thread RNG instances) to ensure predictable behavior
          across concurrent operations.
    """
    def __init__(self, config_file_path: Optional[str] = None, mode: str = 'FULL_V1_2', performance_log_path: str = 'performance_log_FULL_V1_2.csv'):
        self.logger: logging.Logger = logging.getLogger(__name__) # Ensure logger is initialized
        self.mode: str = mode
        self.performance_log_path: str = performance_log_path
        self.lee_instance: Optional['LEE'] = None
        self.available_personas: Dict[str, MarketPersona] = {}
        self.active_persona_name: Optional[str] = None
        self.current_generation: int = 0
        self.mle_instance: Optional[MLE_v0_1] = None
        self.ces_instance: Optional[CES_v1_0] = None
        self.current_mle_bias: Dict[str, Any] = {}
        self.current_ces_vector: Dict[str, Any] = {}
        self.priming_generations: int = 10  # Default, can be overridden by config
        
        self.rng_state_load_path: Optional[str] = None
        self.rng_state_save_path: Optional[str] = None

        # Attributes for the legacy part of SO, to be typed if kept
        self.candidate_slots: List[CandidateStrategySlot] = []
        self.tick_counter: int = 0
        self.regime_classifier: RegimeClassifier = RegimeClassifier()
        self.ab_test_manager: ABTestManager = ABTestManager()
        self.meta_param_monitor: MetaParameterMonitor = MetaParameterMonitor(self) # type: ignore # Pass self if SO is the monitor's target
        self.candidate_regime_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.candidate_pnl_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=SO_SETTINGS.get('ROLLING_PNL_WINDOW_CANDIDATE', 100)))
        self.known_dna: Dict[str, LogicDNA] = {}
        self.dormant_stats: Dict[str, Any] = {} # Placeholder for stats of dormant DNA
        self.tweak_proposals_log: List[Dict[str, Any]] = []
        self.last_logged_correlation: int = 0
        self.last_portfolio_snapshot: int = 0
        self._meta_param_snapshot_counter: int = 0
        self.critical_errors: List[str] = []
        self.config: Optional[Dict[str, Any]] = None # Will hold loaded config
        self.conflict_resolver: Optional[ConflictResolver] = None


        if config_file_path:
            self._load_config(config_file_path)
        else:
            self.logger.warning("SystemOrchestrator initialized without a config file path.")

        if self.lee_instance:
            self.lee_instance.performance_log_path = self.performance_log_path
        if self.mle_instance:
            self.mle_instance.performance_log_path = self.performance_log_path

    def _load_config(self, config_file_path: str) -> None:
        try:
            with open(config_file_path, 'r') as f:
                loaded_config: Dict[str, Any] = json.load(f)
            self.config = loaded_config # Store loaded config
        except FileNotFoundError:
            self.logger.critical(f"Configuration file not found at path: {config_file_path}")
            raise RuntimeError(f"Configuration file not found at path: {config_file_path}")
        except json.JSONDecodeError as e:
            self.logger.critical(f"Error decoding JSON configuration file at path: {config_file_path}. Details: {e}")
            raise RuntimeError(f"Error decoding JSON configuration file at path: {config_file_path}. Details: {e}")

        # --- Validate and initialize LEE ---
        lee_params_config = self.config.get('lee_params')
        if not isinstance(lee_params_config, dict):
            msg = "Config Error: 'lee_params' section is missing or not a dictionary."
            self.logger.critical(msg)
            raise ValueError(msg)

        required_lee_keys_types = {
            'population_size': (int, lambda x: x > 0),
            'mutation_rate_parametric': (float, lambda x: 0.0 <= x <= 1.0),
            'mutation_rate_structural': (float, lambda x: 0.0 <= x <= 1.0),
            'crossover_rate': (float, lambda x: 0.0 <= x <= 1.0),
            'elitism_percentage': (float, lambda x: 0.0 <= x <= 1.0),
            'random_injection_percentage': (float, lambda x: 0.0 <= x <= 1.0),
            'max_depth': (int, lambda x: x > 0),
            'max_nodes': (int, lambda x: x > 0),
            'complexity_weights': (dict, None) # Basic dict check, deeper validation if needed
        }

        for key, (expected_type, condition) in required_lee_keys_types.items():
            value = lee_params_config.get(key)
            if value is None: # Check for missing key first
                msg = f"Config Error: 'lee_params' is missing required key: '{key}'"
                self.logger.critical(msg)
                raise KeyError(msg)
            if not isinstance(value, expected_type):
                msg = f"Config Error: 'lee_params.{key}' must be type {expected_type.__name__}. Found: {type(value).__name__} ({value})"
                self.logger.critical(msg)
                raise ValueError(msg)
            if condition and not condition(value):
                msg = f"Config Error: 'lee_params.{key}' value {value} is not valid (e.g., out of range)."
                self.logger.critical(msg)
                raise ValueError(msg)
        
        self.lee_instance = LEE(**lee_params_config)
        # --- End LEE validation and initialization ---

        personas_config: Dict[str, Dict[str, float]] = self.config.get('personas', {})
        for name, weights in personas_config.items():
            self.available_personas[name] = MarketPersona(name, weights)
        
        initial_persona_key: Optional[str] = self.config.get('initial_active_persona')
        if initial_persona_key and initial_persona_key in self.available_personas:
            self.active_persona_name = initial_persona_key
        elif self.available_personas:
            self.active_persona_name = list(self.available_personas.keys())[0]
        
        if self.lee_instance:
            self.lee_instance.initialize_population() # Assuming seed_dna_templates is optional and handled in LEE
        
        self.current_generation = 0
        self.performance_log_path = self.config.get('performance_log_path', self.performance_log_path) 
        self.mle_instance = MLE_v0_1(self.performance_log_path)
        self.ces_instance = CES_v1_0()
        self.priming_generations = self.config.get('priming_generations', self.priming_generations)
        
        resolver_config = self.config.get('conflict_resolver_config', {})
        self.conflict_resolver = ConflictResolver(resolver_config)

        # Load RNG state if path is provided
        self.rng_state_load_path = self.config.get('rng_state_load_path')
        self.rng_state_save_path = self.config.get('rng_state_save_path')
        if self.rng_state_load_path:
            self.load_rng_state(self.rng_state_load_path) # load_rng_state already handles path validation for loading
        
        # Validate the loaded configuration
        if self.config: # Ensure config is loaded before validation
            self._validate_config(self.config)

        if self.rng_state_save_path:
            # Define a small wrapper function to pass the filepath and a shutdown context
            def _save_rng_on_exit():
                self.logger.info("Attempting to save RNG state on script exit...")
                self.save_rng_state(filepath=self.rng_state_save_path, shutdown_mode=True)

            atexit.register(_save_rng_on_exit)
            self.logger.info(f"RNG state saving registered for exit. Path: {self.rng_state_save_path}")

        self.logger.info(f"SystemOrchestrator initialized. Active persona: {self.active_persona_name}")

    def _validate_config(self, config_data: Dict[str, Any]) -> None:
        """
        Validates critical configuration parameters.
        Raises exceptions if validation fails.
        """
        self.logger.info("Validating configuration...")

        # Validate "lee_params"
        lee_params = config_data.get('lee_params')
        if not isinstance(lee_params, dict):
            msg = "Config Error: 'lee_params' section is missing or not a dictionary."
            self.logger.critical(msg)
            raise ValueError(msg)

        pop_size = lee_params.get('population_size')
        if not isinstance(pop_size, int) or pop_size <= 0:
            msg = f"Config Error: 'lee_params.population_size' must be an integer > 0. Found: {pop_size}"
            self.logger.critical(msg)
            raise ValueError(msg)
        
        max_depth = lee_params.get('max_depth')
        if not isinstance(max_depth, int) or max_depth <= 0:
            msg = f"Config Error: 'lee_params.max_depth' must be an integer > 0. Found: {max_depth}"
            self.logger.critical(msg)
            raise ValueError(msg)
        
        # Validate "personas" existence (basic check)
        personas = config_data.get('personas')
        if not isinstance(personas, dict):
            # Depending on strictness, this could be a warning or an error.
            # For now, let's assume it's critical if LEE needs it for persona-based evolution.
            msg = "Config Error: 'personas' section is missing or not a dictionary."
            self.logger.critical(msg)
            raise ValueError(msg)

        # Validate performance_log_path (already set as self.performance_log_path, but check from config if it's there)
        perf_log_path_conf = config_data.get('performance_log_path')
        if perf_log_path_conf is not None and (not isinstance(perf_log_path_conf, str) or not perf_log_path_conf.strip()):
            msg = f"Config Error: 'performance_log_path' must be a non-empty string if provided. Found: {perf_log_path_conf}"
            self.logger.critical(msg)
            raise ValueError(msg)
        # self.performance_log_path is set from constructor or config, ensure it's valid before use is good,
        # but its existence as a string is implicitly handled by its usage.

        # Validate RNG Paths
        # rng_state_load_path is handled by self.load_rng_state if it exists.
        # We only need to validate its type if present, and its existence.
        rng_load_path = config_data.get('rng_state_load_path')
        if rng_load_path is not None: # If key exists
            if not isinstance(rng_load_path, str) or not rng_load_path.strip():
                msg = f"Config Error: 'rng_state_load_path' must be a non-empty string if provided. Found: {rng_load_path}"
                self.logger.critical(msg)
                raise ValueError(msg)
            if not os.path.exists(rng_load_path): # This check is also in load_rng_state, but good for early validation
                msg = f"Config Error: 'rng_state_load_path' file not found at {rng_load_path}"
                self.logger.critical(msg)
                raise FileNotFoundError(msg)
        
        rng_save_path = config_data.get('rng_state_save_path')
        if rng_save_path is not None: # If key exists
            if not isinstance(rng_save_path, str) or not rng_save_path.strip():
                msg = f"Config Error: 'rng_state_save_path' must be a non-empty string if provided. Found: {rng_save_path}"
                self.logger.critical(msg)
                raise ValueError(msg)
            save_dir = os.path.dirname(rng_save_path) or '.'
            if not os.access(save_dir, os.W_OK):
                msg = f"Config Error: Directory for 'rng_state_save_path' ({save_dir}) is not writable."
                self.logger.critical(msg)
                raise IOError(msg)

        # Validate priming_generations
        priming_gens = config_data.get('priming_generations')
        if priming_gens is not None and (not isinstance(priming_gens, int) or priming_gens < 0):
            msg = f"Config Error: 'priming_generations' must be an integer >= 0. Found: {priming_gens}"
            self.logger.critical(msg)
            raise ValueError(msg)
            
        self.logger.info("Configuration validation completed successfully.")

    def set_active_persona(self, persona_name: str) -> None:
        if persona_name in self.available_personas:
            self.active_persona_name = persona_name
            self.logger.info(f"Active persona set to: {persona_name}")
        else:
            self.logger.error(f"Persona '{persona_name}' not found in available_personas.")
            raise ValueError(f"Persona '{persona_name}' not found in available_personas.")

    def run_n_generations(self, num_generations: int, market_data_snapshots: List[Dict[str, Any]]) -> None:
        if not self.lee_instance or not self.ces_instance or not self.mle_instance or not self.active_persona_name:
            self.logger.error("SystemOrchestrator not fully initialized. LEE, CES, MLE, or active_persona might be missing.")
            return

        for gen_count in range(num_generations):
            phase: str = 'Priming' if self.current_generation < self.priming_generations else 'Active Feedback'
            self.logger.info(f"\n=== Generation {self.current_generation + 1} | Phase: {phase} ===")
            
            current_market_data_snapshot: Dict[str, Any] = market_data_snapshots[gen_count % len(market_data_snapshots)]
            eval_data: Dict[str, Any] = dict(current_market_data_snapshot) 
            eval_data['performance_log_path'] = self.performance_log_path
            eval_data['current_generation'] = self.current_generation + 1

            mle_output_for_ces: Optional[Dict[str, Any]] = None
            current_mle_bias_for_lee: Dict[str, Any]
            
            if phase == 'Priming':
                current_mle_bias_for_lee = {'seed_motifs': {}, 'recommended_operator_biases': {}} 
                current_ces_vector_output: Dict[str, Any] = self.ces_instance.calculate_ces_vector(eval_data, None) 
                if self.mle_instance: 
                    try:
                        self.mle_instance.analyze_recent_performance()
                    except Exception as e:
                        self.logger.warning(f"[Priming] MLE analysis skipped due to error: {e}")
            else: 
                if self.mle_instance:
                    try:
                        current_mle_bias_for_lee = self.mle_instance.analyze_recent_performance()
                        mle_output_for_ces = {'pattern_regime_confidence': current_mle_bias_for_lee.get('some_mle_confidence_metric', 0.0)}
                    except Exception as e:
                        self.logger.warning(f"[Active] MLE analysis failed: {e}. Using neutral bias.")
                        current_mle_bias_for_lee = {'seed_motifs': {}, 'recommended_operator_biases': {}}
                else: 
                     current_mle_bias_for_lee = {'seed_motifs': {}, 'recommended_operator_biases': {}}

                current_ces_vector_output = self.ces_instance.calculate_ces_vector(eval_data, mle_output_for_ces)

            self.current_mle_bias = current_mle_bias_for_lee
            self.current_ces_vector = current_ces_vector_output
            self.logger.info(f"[{phase}] CES Vector: {self.current_ces_vector}")
            if phase == 'Active Feedback': self.logger.info(f"[{phase}] MLE Bias: {self.current_mle_bias}")

            v_ces: float = self.current_ces_vector.get('volatility', 0.5)
            t_ces: float = self.current_ces_vector.get('trend', 0.5)
            l_ces: float = self.current_ces_vector.get('liquidity', 0.5)
            selected_persona_name: str = 'HUNTER_v1' 
            if v_ces > 0.6 and t_ces > 0.6: selected_persona_name = 'HUNTER_v1'
            elif v_ces > 0.7 and l_ces < 0.4: selected_persona_name = 'GUARDIAN_v1'
            
            self.set_active_persona(selected_persona_name)
            self.logger.info(f"[{phase}] Persona selected: {self.active_persona_name}")
            
            active_persona_instance: MarketPersona = self.available_personas[self.active_persona_name]
            
            eval_data_for_lee: Dict[str, Any] = {**eval_data, 'ces_vector': self.current_ces_vector}
            self.lee_instance.run_evolutionary_cycle(active_persona_instance, eval_data_for_lee, self.current_mle_bias, self.current_ces_vector)
            
            self.current_generation += 1
            self.logger.info(f"Generation {self.current_generation} complete. Active persona: {self.active_persona_name}")
        
        if self.rng_state_save_path:
            self.save_rng_state(self.rng_state_save_path)

    def harmonize_influences(self, base_decision_parameters: Dict[str, Any], system_influences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Harmonizes all system influences into a final set of decision parameters.
        Args:
            base_decision_parameters: Dict of base parameters (e.g., fsm_wake_threshold, l1_action_confidence).
            system_influences: Dict of influences from adaptive systems.
        Returns:
            Dict of finalized decision parameters.
        Side effects:
            Logs input and output for traceability.
        """
        self.logger.debug(f"ORCHESTRATOR_HARMONIZE_INPUT|base_params={base_decision_parameters}|influences={system_influences}")
        final_params: Dict[str, Any] = base_decision_parameters.copy()

        if self.config and self.conflict_resolver : 
            if 'fsm_wake_threshold' in final_params:
                base_wake: float = final_params['fsm_wake_threshold']
                modified_wake: float = base_wake
                if 'persona_wake_factor' in system_influences: modified_wake *= system_influences['persona_wake_factor']
                if 'env_score_wake_factor' in system_influences: modified_wake *= system_influences['env_score_wake_factor']
                
                min_thr: float = self.config.get('OrchestratorClamps', {}).get('min_fsm_wake_threshold', 0.00001)
                max_thr: float = self.config.get('OrchestratorClamps', {}).get('max_fsm_wake_threshold', 0.01)
                final_params['fsm_wake_threshold'] = np.clip(modified_wake, min_thr, max_thr) # type: ignore

            if 'l1_action_confidence' in final_params and 'opportunistic_confidence_boost' in system_influences:
                base_conf: float = final_params['l1_action_confidence']
                boost: float = system_influences['opportunistic_confidence_boost']
                modified_conf: float = base_conf + boost
                final_params['l1_action_confidence'] = np.clip(modified_conf, 0.0, 1.0) # type: ignore
            
            if 'anomaly_sequence_suggestion' in system_influences:
                self.logger.info(f"ORCHESTRATOR_SEQUENCE_SUGGESTION|suggestion={system_influences['anomaly_sequence_suggestion']}")

            if self.conflict_resolver.detect_conflicts(final_params, system_influences):
                final_params = self.conflict_resolver.resolve(final_params, system_influences)
        else:
            self.logger.warning("Configuration or ConflictResolver not loaded. Harmonization may be incomplete.")

        self.logger.info(f"ORCHESTRATOR_HARMONIZED_OUTPUT|final_params={final_params}")
        return final_params

    def save_rng_state(self, filepath: str) -> None:
        """
        Saves the current state of the `random` module's RNG to the specified
        filepath using `pickle`. Logs errors if saving fails.
        """
        if not filepath:
            self.logger.warning("RNG state save path not provided. Skipping save.")
            return
        try:
            rng_states = {
                'python_random': random.getstate(),
                'numpy_random': np.random.get_state()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(rng_states, f)
            
            if shutdown_mode:
                self.logger.info(f"Saving RNG state to {filepath} during shutdown.")
            else:
                self.logger.info(f"Dictionary of RNG states (Python, NumPy) saved to {filepath}")
        except IOError as e:
            self.logger.error(f"Error saving RNG states to {filepath}: {e}")
        except pickle.PicklingError as e:
            self.logger.error(f"Error pickling RNG states for saving to {filepath}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while saving RNG state to {filepath}: {e}")

    def load_rng_state(self, filepath: str) -> None:
        """
        Loads the RNG state from the specified filepath using `pickle` and
        applies it to the `random` module. Handles `FileNotFoundError` and
        `pickle.UnpicklingError` gracefully by logging warnings/errors and
        continuing execution.
        """
        if not filepath:
            self.logger.warning("RNG state load path not provided. Skipping load.")
            return
        if not os.path.exists(filepath):
            self.logger.warning(f"RNG state file not found at {filepath}. Skipping load.")
            return
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
            
            if isinstance(loaded_data, dict):
                # New dictionary format
                if 'python_random' in loaded_data:
                    random.setstate(loaded_data['python_random'])
                    self.logger.info("Python's random state loaded from dictionary.")
                else:
                    self.logger.warning("No 'python_random' state found in loaded RNG data dictionary.")
                
                if 'numpy_random' in loaded_data:
                    np.random.set_state(loaded_data['numpy_random'])
                    self.logger.info("NumPy's random state loaded from dictionary.")
                else:
                    self.logger.warning("No 'numpy_random' state found in loaded RNG data dictionary.")
            else:
                # Legacy format: assume it's just Python's random state
                self.logger.warning("Loading legacy RNG state (assumed to be Python's random state only).")
                random.setstate(loaded_data)
                # NumPy's state will remain unchanged or as per its default initialization.
                self.logger.info("Python's random state loaded from legacy format. NumPy state not affected by this legacy file.")

            self.logger.info(f"RNG state loading process completed for {filepath}")
        except IOError as e:
            self.logger.error(f"Error loading RNG state from {filepath}: {e}")
        except pickle.UnpicklingError as e:
            self.logger.error(f"Error unpickling RNG state from {filepath}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading RNG state from {filepath}: {e}")

    def promote_to_candidate_slot(self, graduated_entry: 'PoolDNAEntry') -> None:
        pool_pnl: Any = getattr(graduated_entry, 'rolling_pnl_pool', 'N/A')
        pool_activations: Any = getattr(graduated_entry, 'activation_count_pool', 'N/A')
        promoted_dna_id: str = graduated_entry.dna.id # type: ignore
        promoted_parent_id: Optional[str] = graduated_entry.dna.parent_id # type: ignore

        if len(self.candidate_slots) < MAX_CANDIDATE_SLOTS:
            new_slot: CandidateStrategySlot = CandidateStrategySlot(graduated_entry)
            self.candidate_slots.append(new_slot)
            self.logger.info(
                f"SO: Promoting DNA {promoted_dna_id} to a new candidate slot. "
                f"ParentID={promoted_parent_id}. "
                f"Pool Performance: PnL={pool_pnl}, Activations={pool_activations}."
            )
            self.log_event_snapshot('GRADUATION', new_slot.dna, graduated_entry) # type: ignore
        else:
            worst_slot_idx: int = min(range(len(self.candidate_slots)), key=lambda i: self.candidate_slots[i].pnl_candidate)
            demoted_slot_instance: CandidateStrategySlot = self.candidate_slots[worst_slot_idx]
            demoted_dna_id_str: str = demoted_slot_instance.dna.id # type: ignore
            
            self.logger.info(
                f"SO: Candidate slots full. Demoting worst performing candidate: "
                f"DNA {demoted_dna_id_str}, PnL={demoted_slot_instance.pnl_candidate:.2f}, Sharpe={demoted_slot_instance.sharpe_candidate:.2f}."
            )
            self.log_event_snapshot('DEMOTION', demoted_slot_instance.dna, demoted_slot_instance) # type: ignore
            self.candidate_slots.pop(worst_slot_idx)
            
            new_slot_replacing: CandidateStrategySlot = CandidateStrategySlot(graduated_entry)
            self.candidate_slots.append(new_slot_replacing)
            self.logger.info(
                f"SO: Promoting DNA {promoted_dna_id} to replace demoted candidate {demoted_dna_id_str}. "
                f"ParentID={promoted_parent_id}. "
                f"Pool Performance: PnL={pool_pnl}, Activations={pool_activations}."
            )
            self.log_event_snapshot('GRADUATION', new_slot_replacing.dna, graduated_entry) # type: ignore

    def evaluate_candidates_per_tick(self, market_tick: Dict[str, Any], recent_market_data: Optional[List[Dict[str, Any]]] = None) -> None:
        self.tick_counter += 1
        # Regime classification
        price = market_tick.get('price', None)
        regime = self.regime_classifier.update(price) if price is not None else None
        # Step A/B tests
        ab_finished = self.ab_test_manager.step_all(market_tick)
        if ab_finished:
            for cid, summary in self.ab_test_manager.pop_finished().items():
                # Find candidate slot
                slot = next((s for s in self.candidate_slots if str(s.dna) == cid), None)
                control = summary['control']
                variant = summary['variant']
                promote = False
                ab_sharpe_threshold = self.meta_param_monitor.ab_test_adoption_sharpe_uplift_min if self.meta_param_monitor.enabled else settings.FERAL_CALIBRATOR_SETTINGS['AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN']
                if (variant['sharpe'] > control['sharpe'] * (1 + ab_sharpe_threshold) and
                    variant['pnl'] > control['pnl'] * (1 + AB_PNL_THRESHOLD)):
                    promote = True
                if promote and slot:
                    old_dna = slot.dna
                    slot.dna = self.ab_test_manager.active_tests[cid].variant_dna
                    slot.pnl_history = []  # Reset history for new DNA
                    slot.pnl_candidate = 0.0
                    slot.activation_count_candidate = 0
                    slot.recent_high_sharpe = 0.0
                    slot.last_tweak_implemented_at = self.tick_counter
                    self.logger.info(f"SO AB TEST PROMOTION|Candidate={cid}|Variant adopted as new DNA.\n  Control: PnL={control['pnl']:.2f}, Sharpe={control['sharpe']:.2f}\n  Variant: PnL={variant['pnl']:.2f}, Sharpe={variant['sharpe']:.2f}\n  Reason: Variant outperformed control by >{ab_sharpe_threshold*100:.0f}% Sharpe and >{AB_PNL_THRESHOLD*100:.0f}% PnL. DNA updated. Cooldown started.")
                    # --- Structured log for TWEAK_ADOPTED ---
                    log_event(
                        'TWEAK_ADOPTED',
                        {
                            'candidate_id': cid,
                            'adopted_dna_id': slot.dna.id,
                            'parent_id': getattr(old_dna, 'id', None),
                            'control_result': control,
                            'variant_result': variant,
                            'adopted': True,
                            'changed_params': {param: getattr(slot.dna, param) for param in ['trigger_indicator', 'trigger_operator', 'trigger_threshold', 'action_value']},
                            'system_cpu_load': psutil.cpu_percent(interval=0.1),
                        }
                    )
                    self.log_ab_test_conclusion(cid, control, variant, control['result'], variant['result'], promote, {param: getattr(variant, param) for param in ['trigger_indicator', 'trigger_operator', 'trigger_threshold', 'action_value']})
                else:
                    self.logger.info(f"SO AB TEST COMPLETE|Candidate={cid}|Tweak not adopted.\n  Control: PnL={control['pnl']:.2f}, Sharpe={control['sharpe']:.2f}\n  Variant: PnL={variant['pnl']:.2f}, Sharpe={variant['sharpe']:.2f}\n  Reason: Variant did not significantly outperform control.")
                    # --- Structured log for AB_TEST_CONCLUDED (not adopted) ---
                    log_event(
                        'AB_TEST_CONCLUDED',
                        {
                            'candidate_id': cid,
                            'control_result': control,
                            'variant_result': variant,
                            'adopted': False,
                            'system_cpu_load': psutil.cpu_percent(interval=0.1),
                        }
                    )
        for slot in self.candidate_slots:
            slot.evaluate_tick(market_tick, global_tick=self.tick_counter)
            # Track regime PnL
            dna_id = str(slot.dna)
            if regime:
                self.candidate_regime_pnl[dna_id][str(regime)] += slot.pnl_history[-1] if slot.pnl_history else 0.0
            # Track rolling PnL for correlation
            if slot.pnl_history:
                self.candidate_pnl_history[dna_id].append(slot.pnl_history[-1])
            # Track all known DNA
            self.known_dna[dna_id] = slot.dna
            # Self-calibration
            if slot.self_calibration_needed and recent_market_data is not None:
                slot.run_self_calibration(recent_market_data, self, regime)
        # Prune/demote underperformers
        to_remove = [i for i, slot in enumerate(self.candidate_slots) if slot.ready_for_demotion]
        for idx in reversed(to_remove):
            demoted_slot = self.candidate_slots[idx]
            log_event(
                'CANDIDATE_DEMOTED_OR_PRUNED',
                {
                    'dna_id': demoted_slot.dna.id,
                    'parent_id': demoted_slot.dna.parent_id,
                    'seed_type': demoted_slot.dna.seed_type,
                    'candidate_performance_summary': {
                        'pnl_candidate': demoted_slot.pnl_candidate,
                        'sharpe_candidate': demoted_slot.sharpe_candidate,
                        'activation_count_candidate': demoted_slot.activation_count_candidate,
                        'ticks_in_candidate': demoted_slot.ticks_in_candidate,
                    },
                    'reason': 'demotion_or_pruning',
                    'system_cpu_load': psutil.cpu_percent(interval=0.1),
                }
            )
            self.logger.info(f"SO: Demoting candidate DNA: {self.candidate_slots[idx].dna}")
            self.candidate_slots.pop(idx)
        # Log correlation matrix every 10 cycles
        self.last_logged_correlation += 1
        if self.last_logged_correlation >= 10:
            self.log_correlation_matrix()
            self.last_logged_correlation = 0
        # Log lifecycle advice
        self.log_lifecycle_advice(regime)
        # Portfolio snapshot every 10 cycles
        self.last_portfolio_snapshot += 1
        if self.last_portfolio_snapshot >= 10:
            self.log_portfolio_snapshot(regime)
            self.last_portfolio_snapshot = 0
            self.adjust_influence_multipliers()
            # After portfolio snapshot, increment counter and call meta-param monitor
            self._meta_param_snapshot_counter += 1
            if self._meta_param_snapshot_counter >= getattr(settings, 'META_PARAM_EVAL_EVERY_N_SNAPSHOTS', 5):
                self.meta_param_monitor.evaluate_and_adjust()
                self._meta_param_snapshot_counter = 0
        # Log KPI dashboard every N cycles
        if self.tick_counter % SO_SETTINGS['KPI_DASHBOARD_INTERVAL'] == 0:
            self.log_kpi_dashboard()
        # Analyze Feral Calibrator proposals
        self.analyze_feral_calibrator_proposals()

    def adjust_influence_multipliers(self):
        # Compute fitness scores
        scores = [(slot, self.compute_fitness_score(slot)) for slot in self.candidate_slots]
        if not scores:
            return
        scores_sorted = sorted(scores, key=lambda x: x[1])
        n = len(scores)
        for idx, (slot, score) in enumerate(scores_sorted):
            pct = (idx + 1) / n
            if pct >= INFLUENCE_SCORE_HIGH_PCT:
                slot.influence_multiplier = min(INFLUENCE_MULTIPLIER_MAX, slot.influence_multiplier + INFLUENCE_MULTIPLIER_STEP_UP)
                self.logger.info(f"SO Influence Multiplier UP|{slot.dna}|Fitness={score:.2f}|Multiplier={slot.influence_multiplier:.2f}")
            elif pct <= INFLUENCE_SCORE_LOW_PCT:
                slot.influence_multiplier = max(INFLUENCE_MULTIPLIER_MIN, slot.influence_multiplier - INFLUENCE_MULTIPLIER_STEP_DOWN)
                self.logger.info(f"SO Influence Multiplier DOWN|{slot.dna}|Fitness={score:.2f}|Multiplier={slot.influence_multiplier:.2f}")
            else:
                # No change
                pass

    def get_candidate_confidence_boost(self, trade_direction):
        # Use influence_multiplier for candidates
        for slot in self.candidate_slots:
            if trade_direction == 'buy' and slot.dna.action_target.startswith('buy'):
                return 0.05 * slot.influence_multiplier
            if trade_direction == 'sell' and slot.dna.action_target.startswith('sell'):
                return 0.05 * slot.influence_multiplier
        return 0.0

    def get_portfolio_metrics(self):
        total_pnl = sum(slot.pnl_candidate for slot in self.candidate_slots)
        # Placeholder for drawdown: just min PnL seen so far
        min_pnl = min((min(slot.pnl_history) if slot.pnl_history else 0.0) for slot in self.candidate_slots) if self.candidate_slots else 0.0
        return {'total_candidate_pnl': total_pnl, 'min_drawdown': min_pnl}

    def log_portfolio_metrics(self):
        metrics = self.get_portfolio_metrics()
        self.logger.info(f"SO Portfolio: TotalPnL={metrics['total_candidate_pnl']:.2f} | MinDrawdown={metrics['min_drawdown']:.2f}")

    def log_correlation_matrix(self):
        # Compute rolling correlation matrix for candidate PnL streams
        ids = list(self.candidate_pnl_history.keys())
        if len(ids) < 2:
            return
        try:
            pnl_matrix = np.array([list(self.candidate_pnl_history[dna_id]) for dna_id in ids if len(self.candidate_pnl_history[dna_id]) == self.candidate_pnl_history[ids[0]].maxlen])
            if pnl_matrix.shape[0] < 2:
                return
            corr = np.corrcoef(pnl_matrix)
            self.logger.info(f"SO Portfolio Correlation Matrix|ids={ids}|matrix={np.round(corr,2).tolist()}")
        except Exception as e:
            self.logger.warning(f"SO Portfolio Correlation Matrix|Error computing correlation: {e}")

    def log_lifecycle_advice(self, current_regime):
        # For all known DNA, check regime fit
        for dna_id, dna in self.known_dna.items():
            # Gather stats for this DNA
            regime_stats = self.dormant_stats.get(dna_id, {})
            # If not active, check for activation advice
            if dna_id not in [str(slot.dna) for slot in self.candidate_slots]:
                # If this DNA excelled in current regime, log activation advice
                if current_regime and str(current_regime) in regime_stats and regime_stats[str(current_regime)]['pnl'] > 5:
                    self.logger.info(f"INFO: Dormant Strategy [{dna_id}] shows high potential for current regime {current_regime}. Activation candidate.")
            else:
                # If active, check for deactivation advice
                if current_regime and str(current_regime) in regime_stats and regime_stats[str(current_regime)]['pnl'] < -5:
                    self.logger.warning(f"WARNING: Active Strategy [{dna_id}] historically underperforms in current regime {current_regime}. Deactivation candidate.")

    def log_tweak_proposal(self, candidate_id, orig_params, tweak_params, orig_result, tweak_result, micro_sim_regimes=None, sensitivity_profile=None):
        # Instead of just logging, start A/B test
        self.logger.info(f"SO TWEAK PROPOSAL INITIATED|Candidate={candidate_id}|Orig={orig_params}|Tweak={tweak_params}|Regime={micro_sim_regimes}|Sensitivity={sensitivity_profile}")
        # --- Structured log for TWEAK_PROPOSED ---
        log_event(
            'TWEAK_PROPOSED',
            {
                'candidate_id': candidate_id,
                'orig_params': orig_params,
                'tweak_params': tweak_params,
                'orig_result': orig_result,
                'tweak_result': tweak_result,
                'micro_sim_regimes': micro_sim_regimes,
                'sensitivity_profile': sensitivity_profile,
                'system_cpu_load': psutil.cpu_percent(interval=0.1),
            }
        )
        # --- Structured log for AB_TEST_STARTED ---
        log_event(
            'AB_TEST_STARTED',
            {
                'candidate_id': candidate_id,
                'control_dna': orig_params,
                'variant_dna': tweak_params,
                'tick_started': self.tick_counter,
                'micro_sim_regimes': micro_sim_regimes,
                'system_cpu_load': psutil.cpu_percent(interval=0.1),
            }
        )
        self.ab_test_manager.start_ab_test(candidate_id, LogicDNA(**orig_params), LogicDNA(**tweak_params), self.tick_counter, micro_sim_regimes)
        # Track tweak proposal
        self.tweak_proposals_log.append({'candidate_id': candidate_id, 'tick': self.tick_counter})

    def log_portfolio_snapshot(self, regime):
        lines = [f"\n=== SO PORTFOLIO SNAPSHOT ===",
                 f"Regime: {regime}",
                 f"Num Candidates: {len(self.candidate_slots)}"]
        candidate_details = []
        for slot in self.candidate_slots:
            fitness = self.compute_fitness_score(slot)
            ab_status = 'A/B TESTING' if slot.dna and str(slot.dna) in self.ab_test_manager.active_tests else ''
            lines.append(f"- {slot.dna}: Fitness={fitness:.2f}, PnL={slot.pnl_candidate:.2f}, Sharpe={slot.sharpe_candidate:.2f} {ab_status}")
            candidate_details.append({
                'candidate_id': slot.dna.id,
                'fitness_score': fitness,
                'pnl': slot.pnl_candidate,
                'sharpe': slot.sharpe_candidate,
                'influence_multiplier': getattr(slot, 'influence_multiplier', None),
                'ab_test_status': ab_status,
                'dna_parameters': slot.dna.to_dict(),
            })
        lines.append(f"Portfolio Total PnL: {self.get_portfolio_metrics()['total_candidate_pnl']:.2f}")
        # Shadow advice
        advice = []
        for dna_id, dna in self.known_dna.items():
            regime_stats = self.dormant_stats.get(dna_id, {})
            if dna_id not in [str(slot.dna) for slot in self.candidate_slots]:
                if regime and str(regime) in regime_stats and regime_stats[str(regime)]['pnl'] > 5:
                    advice.append(f"ACTIVATE {dna_id}")
            else:
                if regime and str(regime) in regime_stats and regime_stats[str(regime)]['pnl'] < -5:
                    advice.append(f"DEACTIVATE {dna_id}")
        if advice:
            lines.append("Shadow Advice: " + ", ".join(advice))
        lines.append("==============================\n")
        self.logger.info("\n".join(lines))
        # Structured logging for PORTFOLIO_SNAPSHOT
        log_event(
            'PORTFOLIO_SNAPSHOT',
            {
                'overall_market_regime': regime,
                'system_cpu_load': psutil.cpu_percent(interval=0.1),
                'num_epm_dna': getattr(self, 'epm_pool_size', 'N/A'),
                'num_candidate_strategies': len(self.candidate_slots),
                'candidate_portfolio_pnl_total': self.get_portfolio_metrics()['total_candidate_pnl'],
                'candidate_portfolio_sharpe_avg_total': (sum(slot.sharpe_candidate for slot in self.candidate_slots) / len(self.candidate_slots)) if self.candidate_slots else 0.0,
                'active_candidates_details': candidate_details,
            }
        )

    def compute_fitness_score(self, slot):
        # Sharpe
        sharpe = slot.sharpe_candidate
        # Regime consistency: fraction of regimes with positive PnL
        regime_pnls = self.candidate_regime_pnl.get(str(slot.dna), {})
        num_pos = sum(1 for v in regime_pnls.values() if v > 0)
        regime_consistency = num_pos / max(1, len(regime_pnls))
        # Diversification: 1 - avg corr with other candidates
        my_hist = self.candidate_pnl_history.get(str(slot.dna), [])
        corrs = []
        for other_id, other_hist in self.candidate_pnl_history.items():
            if other_id == str(slot.dna) or len(other_hist) != len(my_hist) or len(my_hist) < 2:
                continue
            corrs.append(np.corrcoef(my_hist, other_hist)[0,1])
        avg_corr = np.mean(corrs) if corrs else 0.0
        diversification = 1 - avg_corr
        # Weighted sum
        score = (FITNESS_WEIGHTS['sharpe'] * sharpe +
                 FITNESS_WEIGHTS['regime_consistency'] * regime_consistency +
                 FITNESS_WEIGHTS['diversification'] * diversification)
        return score

    def run_health_check(self):
        """
        Logs a system health check summary: active DNAs/candidates, CPU load, A/B test status, tweak proposals, and recent critical errors.
        """
        try:
            cpu_load = psutil.cpu_percent(interval=0.5)
        except Exception as e:
            self.logger.warning(f"HealthCheck: Failed to get CPU load: {e}. Using safe default (99.0)")
            cpu_load = 99.0
        num_dnas = len(self.known_dna)
        num_candidates = len(self.candidate_slots)
        ab_tests = list(self.ab_test_manager.active_tests.keys()) if hasattr(self.ab_test_manager, 'active_tests') else []
        num_tweak_proposals = len(self.tweak_proposals_log)
        recent_critical_errors = self.critical_errors[-5:] if hasattr(self, 'critical_errors') else []
        self.logger.info(f"SYSTEM HEALTH CHECK | Active DNAs: {num_dnas} | Candidates: {num_candidates} | CPU Load: {cpu_load:.1f}% | Ongoing A/B Tests: {ab_tests} | Recent Tweak Proposals: {num_tweak_proposals} | Recent Critical Errors: {recent_critical_errors}")
        # --- Structured log for SYSTEM_HEALTH_CHECK ---
        log_event(
            'SYSTEM_HEALTH_CHECK',
            {
                'system_cpu_load': cpu_load,
                'num_dnas': num_dnas,
                'num_candidates': num_candidates,
                'ab_tests': ab_tests,
                'num_tweak_proposals': num_tweak_proposals,
                'recent_critical_errors': recent_critical_errors,
                'timestamp': datetime.now().isoformat(),
            }
        )

    def log_ab_test_conclusion(self, candidate_id, control_dna, variant_dna, control_result, variant_result, adopted, changed_params):
        """
        Logs a detailed snapshot when an A/B test concludes, including lineage and performance.
        """
        lines = [f"\n=== EVENT SNAPSHOT: AB TEST CONCLUSION ===",
                 f"CandidateSlot ID: {candidate_id}",
                 f"Parent DNA ID: {getattr(control_dna, 'id', None)}",
                 f"Tweaked DNA ID: {getattr(variant_dna, 'id', None)}",
                 f"Adopted: {adopted}",
                 f"Changed Params: {changed_params}",
                 f"Control Result: {control_result}",
                 f"Variant Result: {variant_result}",
                 "==============================\n"]
        self.logger.info("\n".join(lines))
        # --- Structured log for AB_TEST_CONCLUDED (adopted) ---
        log_event(
            'AB_TEST_CONCLUDED',
            {
                'candidate_id': candidate_id,
                'control_dna': getattr(control_dna, 'id', None),
                'variant_dna': getattr(variant_dna, 'id', None),
                'control_result': control_result,
                'variant_result': variant_result,
                'adopted': adopted,
                'changed_params': changed_params,
                'system_cpu_load': psutil.cpu_percent(interval=0.1),
            }
        )

    def analyze_feral_calibrator_proposals(self):
        """
        Stub: Logs a conceptual summary of tweak proposal effectiveness (simulated analysis).
        """
        total_proposals = len(self.tweak_proposals_log)
        ab_tests = getattr(self.ab_test_manager, 'active_tests', {})
        # Simulate stats
        ab_successes = getattr(self, 'ab_successes', 0)  # Placeholder
        avg_pnl_uplift = getattr(self, 'avg_pnl_uplift', 0.0)  # Placeholder
        most_common_param = 'trigger_threshold'  # Placeholder
        highest_success_param = 'action_value'  # Placeholder
        lines = [
            "\n=== FERAL CALIBRATOR PROPOSAL ANALYSIS ===",
            f"Total Proposals: {total_proposals}",
            f"A/B Test Success Rate: {ab_successes}/{total_proposals} ({(ab_successes/total_proposals*100) if total_proposals else 0:.1f}%)",
            f"Avg PnL Uplift (Successes): {avg_pnl_uplift:.2f}",
            f"Most Common Param Tweaked: {most_common_param}",
            f"Highest Success Param: {highest_success_param}",
            "==============================\n"
        ]
        self.logger.info("\n".join(lines))

    def log_kpi_dashboard(self):
        """
        Logs a concise, text-based KPI dashboard summarizing system-wide metrics for Peter's review.
        Adds an overall system status line based on CPU load and error counts.
        """
        try:
            cpu_load = psutil.cpu_percent(interval=0.2)
        except Exception as e:
            self.logger.warning(f"KPI Dashboard: Failed to get CPU load: {e}. Using safe default (99.0)")
            cpu_load = 99.0
        # --- System ---
        regime = self.regime_classifier.last_regime if hasattr(self, 'regime_classifier') else None
        num_epm_dnas = getattr(self, 'epm_pool_size', 'N/A')  # Placeholder, can be set externally
        num_candidates = len(self.candidate_slots)
        # --- LEE ---
        num_dnas_generated = getattr(self, 'lee_dnas_generated', 'N/A')  # Placeholder
        mvl_survival_rate = getattr(self, 'lee_mvl_survival_rate', 'N/A')  # Placeholder
        num_tweak_proposals = len(self.tweak_proposals_log)
        ab_tests_active = len(self.ab_test_manager.active_tests) if hasattr(self.ab_test_manager, 'active_tests') else 0
        ab_tests_completed = getattr(self, 'ab_tests_completed', 'N/A')  # Placeholder
        # --- EPM ---
        avg_pool_pnl = getattr(self, 'epm_avg_pool_pnl', 'N/A')  # Placeholder
        num_graduations = getattr(self, 'epm_graduations', 'N/A')  # Placeholder
        num_prunings = getattr(self, 'epm_prunings', 'N/A')  # Placeholder
        # --- Portfolio ---
        total_candidate_pnl = sum(slot.pnl_candidate for slot in self.candidate_slots)
        avg_fitness = (sum(self.compute_fitness_score(slot) for slot in self.candidate_slots) / len(self.candidate_slots)) if self.candidate_slots else 0.0
        tweaks_adopted = getattr(self, 'tweaks_adopted', 'N/A')  # Placeholder
        # --- Errors/Warnings ---
        recent_critical_errors = self.critical_errors[-5:] if hasattr(self, 'critical_errors') else []
        # --- System Status ---
        status = 'NOMINAL'
        if cpu_load > 90 or (recent_critical_errors and len(recent_critical_errors) > 0):
            status = 'WARNING'
        if cpu_load > 98 or (recent_critical_errors and len(recent_critical_errors) > 2):
            status = 'ERROR'
        # --- Dashboard Output ---
        lines = [
            "\n=== KPI DASHBOARD ===",
            f"System: Regime={regime} | CPU={cpu_load:.1f}% | EPM DNAs={num_epm_dnas} | Candidates={num_candidates}",
            f"LEE: DNAsGen={num_dnas_generated} | MVL Survival={mvl_survival_rate} | FeralProposals={num_tweak_proposals} | ABTests Active={ab_tests_active} | ABTests Done={ab_tests_completed}",
            f"EPM: AvgPoolPnL={avg_pool_pnl} | Graduations={num_graduations} | Prunings={num_prunings}",
            f"Portfolio: TotalPnL={total_candidate_pnl:.2f} | AvgFitness={avg_fitness:.2f} | TweaksAdopted={tweaks_adopted}",
            f"Overall System Status: {status}",
        ]
        if recent_critical_errors:
            lines.append(f"CRITICAL ERRORS/WARNINGS: {recent_critical_errors}")
        lines.append("======================\n")
        self.logger.info("\n".join(lines))

    def log_event_snapshot(self, event_type, dna, entry_or_slot):
        """
        Logs a detailed snapshot of a DNA and its state at a key event (graduation, demotion, etc.).
        """
        lines = [f"\n=== EVENT SNAPSHOT: {event_type} ===",
                 f"DNA: {dna}",
                 f"ParentID: {getattr(dna, 'parent_id', None)} | SeedType: {getattr(dna, 'seed_type', None)}"]
        # Add performance summary if available
        if hasattr(entry_or_slot, 'rolling_pnl_pool'):
            lines.append(f"EPM Performance: PnL={entry_or_slot.rolling_pnl_pool}, Activations={entry_or_slot.activation_count_pool}, Ticks={entry_or_slot.ticks_in_pool}")
        if hasattr(entry_or_slot, 'pnl_candidate'):
            lines.append(f"Candidate Performance: PnL={entry_or_slot.pnl_candidate}, Sharpe={entry_or_slot.sharpe_candidate}, Activations={entry_or_slot.activation_count_candidate}, Ticks={entry_or_slot.ticks_in_candidate}")
        lines.append("==============================\n")
        self.logger.info("\n".join(lines)) 