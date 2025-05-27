"""
Main entry point for Project Crypto Bot Peter.
Coordinates the execution of all components and handles CLI interface.
"""

import argparse
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import time
import numpy as np
import os
import threading
from src.live_data_fetcher import WindowsRobustLiveDataManager
import configparser
import sys
from src.preflight import check_environment
from utils.perf import set_perf_enabled, timed
from utils.error_context import ErrorContext, get_current_context
from utils.state_dump import dump_bot_state
from src.adaptive_behavior import (
    MarketPersona, calculate_persona, calculate_env_score, evaluate_anomaly_opportunity, AnomalyPersistenceTracker,
    PersonaTransitionLimiter, EnvScoreSmoothing, AnomalyCircuitBreaker,
    calculate_memory_influenced_penalty, OpportunityActionState, execute_opportunity_action,
    AnomalySequenceDetector, OpportunityActionExecutor, detect_hunter_momentum_burst, AdaptivePenaltyLearner, PersonaEffectivenessTracker, detect_hunter_vwap_momentum_pattern, apply_guardian_tactical_validation, LayeredDecisionArchitecture, EnsembleConfidenceDetector
)
from collections import deque
from utils.run_summary import RunSummaryGenerator
from src.system_orchestrator import SystemOrchestrator
import gc
import json

# Import project modules
from src.data_handling import load_market_data_from_parquet
from src.fsm_sentinel import SleepySentinelFSM
from src.shard_logger import log_raw_shard
from src.feature_engineering import MicroRegimeDetector, RegimeSignal
from src.system_health import MemoryWatchdog
from src.shard_learner import ShardLearner

# --- Robust logging setup: file + console, no duplicates ---
LOG_FILENAME = os.path.abspath('project_crypto_bot_peter.log')
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
formatter = logging.Formatter(log_format)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
# Remove all handlers if re-running interactively
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# File handler
file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent double logging if logger is used directly

# Set DEBUG for all major modules
logging.getLogger('src.fsm_sentinel').setLevel(logging.DEBUG)
logging.getLogger('src.feature_engineering').setLevel(logging.DEBUG)
logging.getLogger('src.shard_learner').setLevel(logging.DEBUG)
logging.getLogger('src.live_data_fetcher').setLevel(logging.DEBUG)

logger.info(f"Logging initialized. File: {LOG_FILENAME}")

# Global configuration
SHARD_FILE_PATH = os.path.join("data", "shards.bin")
FSM_PRICE_THRESHOLD = 0.0005
FSM_EMA_PERIOD = 5
FSM_STABLE_THRESHOLD_PERIODS = 100
FSM_ACTIVE_SLEEP = 0.1
FSM_IDLE_SLEEP = 0.5

# Action mapping
ACTION_MAP = {"HOLD": 0, "BUY": 1, "SELL": 2}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Project Crypto Bot Peter')
    parser.add_argument('--mode', choices=['L0', 'L1'], required=True,
                      help='Operation mode: L0 (monitoring) or L1 (trading)')
    return parser.parse_args()

def get_simulated_price_update(base_price: float = 100.0) -> float:
    """
    Simulate a price update for FSM testing.
    Returns a slightly randomized price around the base.
    """
    return base_price + np.random.normal(0, 0.2)

def log_simulated_trade(action: str, price: float, timestamp: datetime, regime: int) -> None:
    """
    Print a simulated trade log to the console.
    """
    print(f"{timestamp.isoformat()} | SIMULATED TRADE | Action: {action} | Price: {price:.2f} | Regime: {regime}")
    # Optionally, append to a log file here

@timed
def run_level_1_cycle(
    micro_reg_detector: MicroRegimeDetector,
    fsm_instance: SleepySentinelFSM,
    memory_watchdog: Optional[MemoryWatchdog] = None,
    anomaly_report: Optional[Any] = None,
    current_persona: MarketPersona = MarketPersona.GUARDIAN,
    current_env_score: float = 0.5,
    config: Any = None,
    run_summary_gen: Any = None,
    shadow_mode_enabled: bool = True,
    shadow_mode_all_adaptive_logic: bool = True,
    opportunistic_actions_live: bool = False,
    anomaly_persistence_tracker: Any = None,
    opportunity_action_state: Any = None,
    opportunity_action_executor: Any = None,
    adaptive_penalty_learner: Any = None,
    last_active_anomaly_type: str = None,
    last_simulated_pnl: float = None,
    system_orchestrator: Any = None,
    persona_effectiveness_tracker: Any = None
) -> float:
    """
    Run a single Level 1 inference and logging cycle using MicroRegimeDetector.
    Integrates memory-influenced penalty, live/shadow opportunity logic, persona effectiveness, and advanced opportunity executor.
    Args:
        micro_reg_detector: MicroRegimeDetector instance.
        fsm_instance: SleepySentinelFSM instance.
        memory_watchdog: Optional MemoryWatchdog for periodic checks.
        anomaly_report: Optional anomaly report for this cycle.
        current_persona: Current MarketPersona.
        current_env_score: Current environmental score.
        config: Main config object.
        run_summary_gen: RunSummaryGenerator for tracking outcomes.
        shadow_mode_enabled: If True, all adaptive logic is shadowed.
        shadow_mode_all_adaptive_logic: If True, all adaptive logic is shadowed (overrides live actions).
        opportunistic_actions_live: If True, opportunity actions are live (not shadowed).
        anomaly_persistence_tracker: Tracks anomaly persistence for memory-influenced penalties.
        opportunity_action_state: Tracks state for opportunity actions.
        opportunity_action_executor: Executes advanced opportunity logic.
        adaptive_penalty_learner: AdaptivePenaltyLearner for learning penalty adjustments.
        last_active_anomaly_type: Last active anomaly type for learning.
        last_simulated_pnl: Last simulated PnL for learning.
        system_orchestrator: SystemOrchestrator for integrating influences.
        persona_effectiveness_tracker: Tracks persona effectiveness.
    Returns:
        float: Simulated PnL for the current L1 cycle.
    """
    prices, volumes, timestamps = fsm_instance.get_recent_arrays(100)
    micro_reg_detector.update_buffer(prices, volumes, timestamps)
    regime_signal = micro_reg_detector.detect()
    adjusted_confidence = regime_signal.confidence
    # --- Memory-influenced penalty logic ---
    if anomaly_report and anomaly_report.is_anomalous and anomaly_report.confidence_penalty_suggestion and anomaly_report.confidence_penalty_suggestion > 0:
        original_confidence = regime_signal.confidence
        # Use adaptive penalty learner for base penalty
        if adaptive_penalty_learner is not None and hasattr(anomaly_report, 'anomaly_type') and anomaly_report.anomaly_type:
            learned_penalty = adaptive_penalty_learner.get_learned_penalty(anomaly_report.anomaly_type)
            if shadow_mode_all_adaptive_logic:
                logger.info(f"SHADOW_ADAPTIVE_PENALTY|anomaly_type={anomaly_report.anomaly_type}|learned_penalty={learned_penalty:.3f}|default_penalty={anomaly_report.confidence_penalty_suggestion:.3f}")
                penalty = anomaly_report.confidence_penalty_suggestion  # Use default in shadow mode
            else:
                penalty = learned_penalty
        else:
            penalty = anomaly_report.confidence_penalty_suggestion
        # Use memory-influenced penalty
        if anomaly_persistence_tracker is not None:
            penalty = calculate_memory_influenced_penalty(
                penalty, getattr(anomaly_report, 'anomaly_type', None), anomaly_persistence_tracker, config)
        adjusted_confidence = original_confidence * (1.0 - penalty)
        logger.info(f"CONFIDENCE_PENALTY|original={original_confidence:.3f}|penalty={penalty:.3f}|adjusted={adjusted_confidence:.3f}")
    if anomaly_report and anomaly_report.is_anomalous:
        is_critical = (
            (anomaly_report.anomaly_strength is not None and anomaly_report.anomaly_strength > config.getfloat('CrossValidation', 'cross_val_anomaly_strength_threshold', fallback=1.5)) or
            (anomaly_report.anomaly_type in config.get('CrossValidation', 'cross_val_critical_anomaly_types', fallback='MultiFeatureAnomaly,VwmaDeviationAnomaly').split(','))
        )
        if is_critical:
            prev_conf = adjusted_confidence
            adjusted_confidence = adjusted_confidence * config.getfloat('CrossValidation', 'cross_val_confidence_reduction_factor', fallback=0.5)
            logger.warning(f"CROSS-VALIDATION WARNING: Strong RegimeSignal during critical anomaly. Reducing confidence from {prev_conf:.3f} to {adjusted_confidence:.3f}")
    # --- GUARDIAN Persona Tactical Validation Layer ---
    if current_persona == MarketPersona.GUARDIAN and config is not None:
        guardian_validation_multiplier = apply_guardian_tactical_validation(regime_signal, current_env_score, list(anomaly_reports_history), config)
        if not shadow_mode_all_adaptive_logic:
            old_conf = regime_signal.confidence
            regime_signal.confidence = float(np.clip(regime_signal.confidence * guardian_validation_multiplier, 0.0, 1.0))
            logger.info(f"GUARDIAN_TACTICAL_VALIDATION|multiplier={guardian_validation_multiplier:.3f}|old_conf={old_conf:.3f}|new_conf={regime_signal.confidence:.3f}")
        else:
            would_be_conf = float(np.clip(regime_signal.confidence * guardian_validation_multiplier, 0.0, 1.0))
            logger.info(f"GUARDIAN_TACTICAL_VALIDATION_SHADOW|multiplier={guardian_validation_multiplier:.3f}|current_conf={regime_signal.confidence:.3f}|would_be_conf={would_be_conf:.3f}")
    # --- Lazy Evaluation Gate for Opportunity Logic ---
    lazy_gate_max_env_score = config.getfloat('Opportunity', 'lazy_gate_max_env_score', fallback=0.4) if config is not None else 0.4
    opportunity_eval_result = None
    boosted_confidence = None
    should_evaluate_opportunity = (
        current_persona == MarketPersona.HUNTER and current_env_score < lazy_gate_max_env_score
    )
    if should_evaluate_opportunity and config is not None and anomaly_report is not None:
        opportunity_eval_result = evaluate_anomaly_opportunity(
            anomaly_report, current_persona, current_env_score, config, base_regime_confidence=regime_signal.confidence)
        logger.info(f"OPPORTUNITY_SIGNAL|type={opportunity_eval_result.get('opportunity_type')}|would_boost_by={opportunity_eval_result.get('would_boost_confidence_by', 0.0):.3f}|original_conf={opportunity_eval_result.get('original_confidence', 0.0):.3f}|shadowed={opportunity_eval_result.get('action_taken_shadowed')}|persona={current_persona.name}")
        if run_summary_gen is not None:
            run_summary_gen.log_opportunity_identified(opportunity_eval_result)
        # --- Live/shadow opportunity action logic ---
        if opportunity_eval_result.get('opportunity_type') != 'none':
            if not shadow_mode_all_adaptive_logic and opportunistic_actions_live:
                # Actually execute the opportunity action
                boosted_confidence = execute_opportunity_action(
                    opportunity_eval_result, regime_signal, config, opportunity_action_state)
                if boosted_confidence is not None:
                    logger.info(f"OPPORTUNITY_APPLIED|boosted_confidence={boosted_confidence:.3f}")
                    adjusted_confidence = boosted_confidence
            else:
                # Shadow mode: log what would happen
                boosted_confidence = execute_opportunity_action(
                    opportunity_eval_result, regime_signal, config, opportunity_action_state)
                logger.info(f"SHADOW_OPPORTUNITY_ACTION|Would boost confidence to {boosted_confidence}, but shadow/disabled.")
    # --- Advanced Opportunity Executor (Phase 2) ---
    if should_evaluate_opportunity and opportunity_action_executor is not None and opportunity_eval_result is not None:
        base_params = {'base_confidence': regime_signal.confidence, 'env_score': current_env_score, 'size': 1.0, 'tp': 1.0, 'sl': 1.0}
        adv_result = opportunity_action_executor.execute_advanced_opportunity(
            opportunity_eval_result, base_params, current_persona, config)
        logger.info(f"OPPORTUNITY_ADV_RESULT|result={adv_result}")
    # --- Orchestrator for L1 action confidence ---
    base_confidence = regime_signal.confidence
    system_influences = {}
    if opportunity_eval_result and opportunity_eval_result.get('would_boost_confidence_by', 0.0) > 0:
        system_influences['opportunistic_confidence_boost'] = opportunity_eval_result.get('would_boost_confidence_by', 0.0)
    # Add other influences as needed (e.g., persona, anomaly sequence, etc.)
    base_decision_parameters = {'l1_action_confidence': base_confidence}
    orchestrated_params = system_orchestrator.harmonize_influences(base_decision_parameters, system_influences) if system_orchestrator else base_decision_parameters
    logger.info(f"ORCHESTRATOR_APPLIED|inputs={base_decision_parameters}|influences={system_influences}|output={orchestrated_params}")
    clamped_confidence = orchestrated_params['l1_action_confidence'] if 'l1_action_confidence' in orchestrated_params else base_confidence
    if clamped_confidence != adjusted_confidence:
        logger.warning(f"PARAM_CLAMP|confidence|from={adjusted_confidence:.3f}|to={clamped_confidence:.3f}|min=0.0|max=1.0")
    regime_signal.confidence = clamped_confidence
    fsm_instance.set_regime_signal(regime_signal)
    action_str = "HOLD"
    if regime_signal.trend_strength > 0.6 and regime_signal.volatility_regime in ["low", "medium"] and clamped_confidence > 0.5:
        action_str = "BUY"
    elif regime_signal.trend_strength < -0.6 and regime_signal.volatility_regime in ["low", "medium"] and clamped_confidence > 0.5:
        action_str = "SELL"
    action_int = ACTION_MAP.get(action_str, ACTION_MAP["HOLD"])
    regime_map = {"low": 0, "medium": 1, "high": 2, "unknown": 3}
    regime_byte = regime_map.get(regime_signal.volatility_regime, 3)
    calculated_volatility_value = getattr(micro_reg_detector.vol_regime, 'last_raw_volatility', 0.0) or 0.0
    latest_update = fsm_instance.latest
    log_raw_shard(
        SHARD_FILE_PATH,
        int(timestamps[-1]) if len(timestamps) > 0 else int(time.time()),
        float(prices[-1]) if len(prices) > 0 else 0.0,
        fsm_instance.current_ema if fsm_instance.current_ema is not None else 0.0,
        calculated_volatility_value,
        regime_byte,
        action_int,
        0.0,
        0,
        0,
        latest_update.volume if latest_update and hasattr(latest_update, 'volume') else 0.0,
        0.0,
        float(regime_signal.vwma_deviation_pct),
        int(getattr(latest_update, 'fetch_latency_ms', 0)) if latest_update else 0,
        int(getattr(latest_update, 'latency_spike_flag', 0)) if latest_update else 0,
        float(getattr(latest_update, 'latency_volatility_index', 0.0)) if latest_update else 0.0
    )
    print(f"Level 1: Trend={regime_signal.trend_strength:.2f}, VolRegime={regime_signal.volatility_regime}, Conf={clamped_confidence:.2f}, Action={action_str}, Price={prices[-1] if len(prices) > 0 else 0.0:.2f}")
    # --- Persona Effectiveness/Simulated PnL for Run Summary ---
    simulated_pnl = 0.0
    # MVP: +1 if BUY and trend_strength > 0.6, -1 if BUY and trend_strength < -0.6, +1 if SELL and trend_strength < -0.6, -1 if SELL and trend_strength > 0.6, 0 for HOLD or low confidence
    if action_str == "BUY":
        if regime_signal.trend_strength > 0.6:
            simulated_pnl = 1.0
        elif regime_signal.trend_strength < -0.6:
            simulated_pnl = -1.0
    elif action_str == "SELL":
        if regime_signal.trend_strength < -0.6:
            simulated_pnl = 1.0
        elif regime_signal.trend_strength > 0.6:
            simulated_pnl = -1.0
    # Log SHADOW_TRADE_DECISION marker
    opportunity_involved = (opportunity_eval_result is not None and opportunity_eval_result.get('opportunity_type', 'none') != 'none')
    logger.info(f"SHADOW_TRADE_DECISION|action={action_str}|confidence={clamped_confidence:.3f}|persona={current_persona.name}|env_score={current_env_score:.3f}|opportunity_involved={opportunity_involved}")
    if run_summary_gen is not None:
        run_summary_gen.track_l1_outcome(current_persona.name, simulated_pnl, regime_signal.confidence, current_env_score, action_str)
        # New: log shadowed opportunity outcome for post-hoc analysis
        if opportunity_involved:
            run_summary_gen.log_shadowed_opportunity_outcome(opportunity_eval_result['opportunity_type'], opportunity_eval_result, simulated_pnl)
    # --- AdaptivePenaltyLearner update after L1 cycle ---
    if adaptive_penalty_learner is not None:
        # Use anomaly_report.anomaly_type if present and is_anomalous
        anomaly_type_for_learning = None
        if anomaly_report and getattr(anomaly_report, 'is_anomalous', False) and getattr(anomaly_report, 'anomaly_type', None):
            anomaly_type_for_learning = anomaly_report.anomaly_type
        if anomaly_type_for_learning:
            adaptive_penalty_learner.update_profile_with_l1_outcome(anomaly_type_for_learning, simulated_pnl)
    if memory_watchdog is not None:
        memory_watchdog.periodic_check()
    # --- Reset opportunity boost state after this cycle ---
    if opportunity_action_state is not None:
        opportunity_action_state.opportunity_active = None
        opportunity_action_state.consecutive_actions = 0
    # --- HUNTER Persona Fast Pattern Detection (NumPy - Shadowed) ---
    if current_persona == MarketPersona.HUNTER and config is not None:
        price_np = np.array(prices, dtype=np.float32)
        if detect_hunter_momentum_burst(price_np, config):
            logger.info(f"HUNTER_MOMENTUM_BURST|Detected in run_level_1_cycle|persona={current_persona.name}")
        # --- VWAP Momentum Pattern Detection ---
        current_price = prices[-1] if len(prices) > 0 else 0.0
        vwap_or_vwma = micro_reg_detector._calc_vwma()[0] if hasattr(micro_reg_detector, '_calc_vwma') else 0.0
        price_velocity_1min = (current_price - prices[-60]) / prices[-60] if len(prices) >= 61 and prices[-60] != 0 else 0.0
        volume_ratio_1min_vs_5min = (sum(volumes[-60:]) / (sum(volumes[-300:]) / 5)) if len(volumes) >= 300 and sum(volumes[-300:]) > 0 else 0.0
        vwap_pattern_signal = detect_hunter_vwap_momentum_pattern(current_price, vwap_or_vwma, price_velocity_1min, volume_ratio_1min_vs_5min, config)
        if vwap_pattern_signal is not None:
            logger.info(f"HUNTER_INSTINCT_PATTERN|type=vwap_momentum|direction={vwap_pattern_signal['direction']}|calculated_confidence={vwap_pattern_signal['calculated_confidence']:.3f}")
            hunter_vwap_momentum_live = config.getboolean('MarketPersona', 'hunter_vwap_momentum_live', fallback=False)
            boost_factor = config.getfloat('HunterPatterns', 'vwap_momentum_confidence_boost_factor', fallback=0.1)
            confidence_boost = vwap_pattern_signal['calculated_confidence'] * boost_factor
            if hunter_vwap_momentum_live and not shadow_mode_all_adaptive_logic:
                old_conf = regime_signal.confidence
                regime_signal.confidence = float(np.clip(regime_signal.confidence + confidence_boost, 0.0, 1.0))
                logger.info(f"HUNTER_VWAP_MOMENTUM_LIVE|old_conf={old_conf:.3f}|boost={confidence_boost:.3f}|new_conf={regime_signal.confidence:.3f}")
            else:
                logger.info(f"HUNTER_VWAP_MOMENTUM_SHADOW|would_boost={confidence_boost:.3f}|current_conf={regime_signal.confidence:.3f}")
    # --- Persona Effectiveness Tracker update after L1 cycle ---
    if persona_effectiveness_tracker is not None:
        persona_effectiveness_tracker.record_l1_outcome(current_persona, current_env_score, simulated_pnl)
    # --- Layered Decision Architecture & Ensemble Confidence Detector (MVP integration) ---
    # Instantiate engines (could be moved to main if persistent state needed)
    layered_decision_engine = LayeredDecisionArchitecture(config)
    ensemble_confidence_detector = EnsembleConfidenceDetector(config)
    # Prepare market state snapshot for layers/ensemble
    market_state_snapshot = {
        'current_price': prices[-1] if len(prices) > 0 else None,
        'env_score': current_env_score,
        'persona': current_persona,
        'regime_signal': regime_signal,
        'anomaly_report': anomaly_report,
        'opportunity_eval_result': opportunity_eval_result,
        'recent_anomaly_reports': list(anomaly_reports_history),
        'volumes': volumes,
        'timestamps': timestamps,
        # Add more as needed for future layers
    }
    # Call LayeredDecisionArchitecture (MVP: log or apply confidence_mod)
    regime_signal = layered_decision_engine.process_l1_decision(market_state_snapshot, current_persona, regime_signal, shadow_mode=shadow_mode_all_adaptive_logic)
    # Call EnsembleConfidenceDetector (MVP: log only)
    super_confidence_signal = ensemble_confidence_detector.detect_super_confidence_conditions({
        'current_persona': current_persona,
        'regime_signal': regime_signal,
        'env_score': current_env_score,
        'anomaly_report': anomaly_report,
        'opportunity_eval_result': opportunity_eval_result,
        # Add more as needed
    })
    return simulated_pnl

def validate_config(config):
    errors = []
    warnings = []
    # FSM
    fsm_cfg = config['fsm'] if 'fsm' in config else {}
    try:
        thr = float(fsm_cfg.get('price_deviation_threshold', 0.0005))
        if not (0 < thr < 0.1):
            errors.append(f"[fsm] price_deviation_threshold out of range: {thr}")
        min_conf = float(fsm_cfg.get('min_confidence_for_l1_trigger', 0.5))
        if not (0 <= min_conf <= 1):
            errors.append(f"[fsm] min_confidence_for_l1_trigger out of range: {min_conf}")
    except Exception as e:
        errors.append(f"[fsm] Error parsing critical parameters: {e}")
    # DefensiveMeasures
    def_cfg = config['DefensiveMeasures'] if 'DefensiveMeasures' in config else {}
    try:
        factor = float(def_cfg.get('latency_volatility_defensive_fsm_wake_factor', 1.5))
        if factor < 1.0:
            warnings.append(f"[DefensiveMeasures] latency_volatility_defensive_fsm_wake_factor < 1.0 (should be >= 1.0): {factor}")
    except Exception as e:
        errors.append(f"[DefensiveMeasures] Error parsing latency_volatility_defensive_fsm_wake_factor: {e}")
    # CrossValidation
    cross_cfg = config['CrossValidation'] if 'CrossValidation' in config else {}
    try:
        red = float(cross_cfg.get('cross_val_confidence_reduction_factor', 0.5))
        if not (0 < red < 1):
            errors.append(f"[CrossValidation] cross_val_confidence_reduction_factor out of range: {red}")
    except Exception as e:
        errors.append(f"[CrossValidation] Error parsing cross_val_confidence_reduction_factor: {e}")
    # AnomalyDetection
    anom_cfg = config['AnomalyDetection'] if 'AnomalyDetection' in config else {}
    try:
        for k in ['anomaly_penalty_default_latency', 'anomaly_penalty_default_vwma_deviation', 'anomaly_penalty_default_volatility', 'anomaly_penalty_default_multi_feature']:
            v = float(anom_cfg.get(k, 0.1))
            if not (0 <= v < 1):
                errors.append(f"[AnomalyDetection] {k} out of range: {v}")
    except Exception as e:
        errors.append(f"[AnomalyDetection] Error parsing penalty values: {e}")
    # Latency
    lat_cfg = config['Latency'] if 'Latency' in config else {}
    try:
        win = int(lat_cfg.get('latency_volatility_window_size', 20))
        if win < 2:
            errors.append(f"[Latency] latency_volatility_window_size too small: {win}")
    except Exception as e:
        errors.append(f"[Latency] Error parsing latency_volatility_window_size: {e}")
    # Log all
    for w in warnings:
        logger.warning(w)
    for e in errors:
        logger.error(e)
    if errors:
        logger.error("Critical configuration errors detected. Aborting startup.")
        sys.exit(1)

def main() -> None:
    """
    Main entry point for the trading bot.
    Failure Modes: Aborts on failed preflight, config, or critical errors. Logs all context on crash.
    Performance Notes: Main loop is I/O and sleep bound; L1 cycles and anomaly detection are the main CPU cost.
    """
    check_environment()
    args = parse_args()
    logger.info(f"Starting Project Crypto Bot Peter in {args.mode} mode")
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Set perf monitor flag
    perf_enabled = config.getboolean('Performance', 'perf_monitor_enabled', fallback=False)
    set_perf_enabled(perf_enabled)
    # Set up state dump interval
    state_dump_interval = config.getint('Debugging', 'state_dump_check_interval_seconds', fallback=30)
    # Shadow mode flag
    shadow_mode_enabled = config.getboolean('AdaptiveBehavior', 'shadow_mode_enabled', fallback=True)
    # Run summary generator
    run_summary_gen = RunSummaryGenerator(config)
    # Set up global error context logging
    def custom_excepthook(exc_type, exc_val, exc_tb):
        import traceback
        context = get_current_context()
        tb_str = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
        logger.critical(f"CRASH|CONTEXT_DICT={repr(context)}|TRACEBACK={tb_str}")
        sys.__excepthook__(exc_type, exc_val, exc_tb)
    sys.excepthook = custom_excepthook
    # Initialize FSM and MicroRegimeDetector
    fsm = SleepySentinelFSM(
        threshold=FSM_PRICE_THRESHOLD,
        ema_period=FSM_EMA_PERIOD,
        stable_threshold_periods=FSM_STABLE_THRESHOLD_PERIODS,
        active_sleep=FSM_ACTIVE_SLEEP,
        idle_sleep=FSM_IDLE_SLEEP
    )
    micro_regime_detector = MicroRegimeDetector()
    memory_watchdog = MemoryWatchdog()
    shard_learner = ShardLearner(SHARD_FILE_PATH)
    def fsm_price_update_callback(update):
        fsm.add_price_volume_update(update)
    print("Bot initialized. FSM created. MicroRegimeDetector loaded.")
    # Live data manager setup
    live_data_manager = WindowsRobustLiveDataManager(symbol="BTCUSDT", fetch_interval=1.0)
    live_data_manager.set_update_callback(fsm_price_update_callback)
    live_data_manager.start()
    validate_config(config)
    defensive_cfg = config['DefensiveMeasures'] if 'DefensiveMeasures' in config else {}
    latency_vol_circuit_breaker_thr = float(defensive_cfg.get('latency_volatility_circuit_breaker_threshold', 0.8))
    latency_vol_defensive_factor = float(defensive_cfg.get('latency_volatility_defensive_fsm_wake_factor', 1.5))
    crossval_cfg = config['CrossValidation'] if 'CrossValidation' in config else {}
    cross_val_anomaly_strength_threshold = float(crossval_cfg.get('cross_val_anomaly_strength_threshold', 1.5))
    cross_val_critical_anomaly_types = [s.strip() for s in crossval_cfg.get('cross_val_critical_anomaly_types', 'MultiFeatureAnomaly,VwmaDeviationAnomaly').split(',')]
    cross_val_confidence_reduction_factor = float(crossval_cfg.get('cross_val_confidence_reduction_factor', 0.5))
    # --- Adaptive Intelligence State ---
    current_persona = MarketPersona.GUARDIAN
    current_env_score = 0.5
    persona_update_interval = 5  # L0 polls
    env_score_update_interval = 5  # L0 polls
    anomaly_persistence_tracker = AnomalyPersistenceTracker(
        persistence_check_interval=config.getint('AnomalyPersistence', 'persistence_check_interval_seconds', fallback=30),
        anomaly_timeout=config.getint('AnomalyPersistence', 'anomaly_timeout_seconds', fallback=300),
        config=config
    )
    opportunity_action_state = OpportunityActionState()
    anomaly_reports_history = deque(maxlen=30)
    latency_values_history = deque(maxlen=60)
    regime_trend_history = deque(maxlen=30)
    persona_update_counter = 0
    env_score_update_counter = 0
    persistence_log_counter = 0
    persona_transition_limiter = PersonaTransitionLimiter(config)
    env_score_smoother = EnvScoreSmoothing(config)
    anomaly_circuit_breaker = AnomalyCircuitBreaker(config)
    anomaly_sequence_detector = AnomalySequenceDetector(config['AnomalySequences'] if 'AnomalySequences' in config else {}, config)
    opportunity_action_executor = OpportunityActionExecutor(config)
    adaptive_penalty_learner = AdaptivePenaltyLearner(config)
    system_orchestrator = SystemOrchestrator(config)
    persona_effectiveness_tracker = PersonaEffectivenessTracker(config)
    if args.mode == "L0":
        print("Starting in Level 0 (Sentinel Mode)...")
        fsm.force_level(0)
        last_dump_check = time.time()
        pnl_from_previous_l1 = 0.0  # Stores the PnL from the previous L1 cycle for context
        simulated_pnl = 0.0         # Stores the PnL from the current L1 cycle
        try:
            with ErrorContext({'mode': 'L0', 'fsm_state': repr(getattr(fsm, 'state', None))}):
                while True:
                    now = time.time()
                    run_summary_gen.log_l0_poll()
                    latest_price = fsm.get_latest_price
                    latest_timestamp = fsm.get_latest_timestamp
                    ema_display = f"{fsm.current_ema:.2f}" if fsm.current_ema is not None else "N/A"
                    print(f"L0 Polling: Latest price in FSM: {latest_price if latest_price is not None else 'N/A'} | EMA={ema_display} | StablePeriods={fsm.market_stable_periods_count}")
                    latest_update = fsm.latest
                    # --- Collect rolling histories for adaptive intelligence ---
                    if latest_update and hasattr(latest_update, 'fetch_latency_ms'):
                        latency_values_history.append(getattr(latest_update, 'fetch_latency_ms', 0.0))
                    if getattr(fsm, 'latest_regime_signal', None) is not None:
                        regime_trend_history.append(getattr(fsm.latest_regime_signal, 'trend_strength', 0.0))
                    # --- Anomaly detection and persistence tracking ---
                    anomaly_report = None
                    if anomaly_circuit_breaker.should_process_anomaly(now):
                        anomaly_report = shard_learner.detect_anomalies()
                        anomaly_reports_history.append(anomaly_report)
                        anomaly_persistence_tracker.update(anomaly_report, now)
                        if anomaly_report and getattr(anomaly_report, 'is_anomalous', False):
                            anomaly_circuit_breaker.record_anomaly_detected(now)
                            logger.info(f"ANOMALY_DETECTED|type={getattr(anomaly_report, 'anomaly_type', 'N/A')}|strength={getattr(anomaly_report, 'anomaly_strength', 0.0):.3f}|penalty_suggested={getattr(anomaly_report, 'confidence_penalty_suggestion', 0.0):.3f}")
                            run_summary_gen.log_anomaly_detected(getattr(anomaly_report, 'anomaly_type', 'N/A'), current_env_score)
                    else:
                        logger.warning("ANOMALY_CIRCUIT_OPEN|Anomaly processing temporarily suspended due to high detection rate.")
                        run_summary_gen.log_circuit_breaker_trigger()
                    # --- Anomaly Sequence Detection ---
                    detected_sequence = anomaly_sequence_detector.check_sequence_completion(anomaly_persistence_tracker.anomaly_history) if hasattr(anomaly_sequence_detector, 'check_sequence_completion') else None
                    if detected_sequence:
                        logger.info(f"ANOMALY_SEQUENCE_DETECTED|pattern={detected_sequence['sequence']}|confidence={detected_sequence['confidence']:.2f}")
                        suggested_response = anomaly_sequence_detector.execute_sequence_response(detected_sequence) if hasattr(anomaly_sequence_detector, 'execute_sequence_response') else None
                        logger.info(f"ANOMALY_SEQUENCE_RESPONSE_SUGGESTED|action={suggested_response['action']}|details={suggested_response}")
                    # --- Periodic persona and env_score updates ---
                    persona_update_counter += 1
                    env_score_update_counter += 1
                    persistence_log_counter += 1
                    # --- Calculate regime_stability MVP ---
                    regime_stability_max = config.getfloat('MarketPersona', 'regime_stability_max_stable_periods', fallback=200.0)
                    if getattr(fsm, 'latest_regime_signal', None) is not None and hasattr(fsm.latest_regime_signal, 'confidence'):
                        regime_stability = fsm.latest_regime_signal.confidence
                    else:
                        regime_stability = min(fsm.market_stable_periods_count / regime_stability_max, 1.0)
                    # --- Calculate env_score first using previous persona ---
                    if env_score_update_counter >= env_score_update_interval:
                        raw_env_score = calculate_env_score(list(anomaly_reports_history), list(latency_values_history), list(regime_trend_history), current_persona, config)
                        current_env_score = env_score_smoother.update(raw_env_score)
                        run_summary_gen.log_env_score((current_env_score, now))
                        logger.info(f"ENV_SCORE_UPDATE|new_score={current_env_score:.3f}")
                        env_score_update_counter = 0
                    # --- Then calculate new persona using new env_score ---
                    if persona_update_counter >= persona_update_interval:
                        new_persona = calculate_persona(list(anomaly_reports_history)[-10:], float(getattr(latest_update, 'latency_volatility_index', 0.0)), regime_stability, current_env_score, config)
                        if new_persona != current_persona:
                            smoothed_env_score = current_env_score
                            score_gap = abs(current_env_score - (env_score_smoother.smoothed_score if hasattr(env_score_smoother, 'smoothed_score') else current_env_score))
                            if not persona_transition_limiter.can_transition(now, current_persona, new_persona):
                                logger.info(f"PERSONA_TRANSITION_BLOCKED|current_persona={current_persona.name}|intended_persona={new_persona.name}|env_score={current_env_score:.3f}|smoothed_env_score={smoothed_env_score:.3f}|score_gap={score_gap:.3f}")
                                run_summary_gen.log_persona_transition_blocked()
                            else:
                                logger.info(f"PERSONA_TRANSITION|from={current_persona.name}|to={new_persona.name}|trigger=adaptive_logic|env_score={current_env_score:.3f}")
                                run_summary_gen.log_persona_transition(current_persona.name, new_persona.name, "adaptive_logic", current_env_score)
                                persona_transition_limiter.record_transition(now, current_persona, new_persona)
                                current_persona = new_persona
                        persona_update_counter = 0
                    if persistence_log_counter >= anomaly_persistence_tracker.persistence_check_interval:
                        logger.info(f"ANOMALY_PERSISTENCE_SUMMARY|summary={anomaly_persistence_tracker.get_active_anomaly_summary()}")
                        persistence_log_counter = 0
                    # --- Persona/env_score-based FSM wake threshold (now via orchestrator) ---
                    base_fsm_wake_threshold = config.getfloat('fsm', 'price_deviation_threshold', fallback=0.0005)
                    persona_modifier = 1.0
                    if current_persona == MarketPersona.HUNTER:
                        persona_modifier = config.getfloat('MarketPersona', 'hunter_wake_sensitivity_factor', fallback=0.8)
                    elif current_persona == MarketPersona.GHOST:
                        persona_modifier = config.getfloat('MarketPersona', 'ghost_wake_sensitivity_factor', fallback=2.0)
                        logger.info("GHOST persona active, L1 wake checks may be suppressed or highly conservative.")
                    env_score_wake_factor = 1.0 + (config.getfloat('EnvScore', 'env_score_fsm_wake_impact_factor', fallback=1.0) * current_env_score)
                    latency_circuit_breaker_modifier = 1.0
                    if latest_update:
                        latest_lvi = float(getattr(latest_update, 'latency_volatility_index', 0.0))
                        if latest_lvi > latency_vol_circuit_breaker_thr:
                            latency_circuit_breaker_modifier = latency_vol_defensive_factor
                            logger.warning(f"DEFENSIVE_STANCE_TRIGGERED|High Latency Volatility|LVI={latest_lvi:.3f}|threshold={latency_vol_circuit_breaker_thr}")
                    # Gather influences for orchestrator
                    base_decision_parameters = {'fsm_wake_threshold': base_fsm_wake_threshold}
                    system_influences = {
                        'persona_wake_factor': persona_modifier,
                        'env_score_wake_factor': env_score_wake_factor,
                        'latency_circuit_breaker_factor': latency_circuit_breaker_modifier
                    }
                    # Add anomaly sequence suggestion if available (MVP: log only)
                    # system_influences['anomaly_sequence_suggestion'] = ...
                    orchestrated_params = system_orchestrator.harmonize_influences(base_decision_parameters, system_influences)
                    logger.info(f"ORCHESTRATOR_APPLIED|inputs={base_decision_parameters}|influences={system_influences}|output={orchestrated_params}")
                    # --- SHADOW MODE LOGIC ---
                    if shadow_mode_enabled:
                        logger.info(f"SHADOW_MODE_WAKE_THRESHOLD|current_persona={current_persona.name}|current_env_score={current_env_score:.3f}|base_threshold={base_fsm_wake_threshold:.6f}|shadow_effective_threshold={orchestrated_params['fsm_wake_threshold']:.6f}")
                        effective_wake_threshold_for_fsm = base_fsm_wake_threshold * latency_circuit_breaker_modifier
                    else:
                        effective_wake_threshold_for_fsm = orchestrated_params['fsm_wake_threshold']
                    logger.info(f"FSM_WAKE_THRESHOLD|base={base_fsm_wake_threshold:.6f}|persona_mod={persona_modifier:.2f}|env_score_mod={env_score_wake_factor:.2f}|latency_mod={latency_circuit_breaker_modifier:.2f}|effective={effective_wake_threshold_for_fsm:.6f}")
                    if fsm.check_for_wake_trigger(threshold_override=effective_wake_threshold_for_fsm):
                        dev_pct = 0.0
                        if fsm.current_ema and latest_price:
                            dev_pct = abs((latest_price - fsm.current_ema) / fsm.current_ema)
                        logger.info(f"FSM_WAKE_TRIGGERED|price={latest_price}|ema={fsm.current_ema}|deviation_pct={dev_pct:.4f}|threshold={effective_wake_threshold_for_fsm:.4f}|persona={current_persona.name}|env_score={current_env_score:.3f}")
                        print(f"FSM WAKE TRIGGER! Current Price: {latest_price}, EMA: {fsm.current_ema}. Simulating Level 1 cycle.")
                        with ErrorContext({'mode': 'L1', 'regime_signal': repr(getattr(fsm, 'latest_regime_signal', None)), 'anomaly_report': repr(anomaly_report)}):
                            run_summary_gen.log_l1_cycle()
                            # Call L1 and get its actual PnL, passing previous L1 PnL for context
                            simulated_pnl = run_level_1_cycle(
                                micro_regime_detector,
                                fsm,
                                memory_watchdog,
                                anomaly_report,
                                current_persona,
                                current_env_score,
                                config,
                                run_summary_gen=run_summary_gen,
                                shadow_mode_enabled=shadow_mode_enabled,
                                shadow_mode_all_adaptive_logic=shadow_mode_enabled,
                                opportunistic_actions_live=shadow_mode_enabled,
                                anomaly_persistence_tracker=anomaly_persistence_tracker,
                                opportunity_action_state=opportunity_action_state,
                                opportunity_action_executor=opportunity_action_executor,
                                adaptive_penalty_learner=adaptive_penalty_learner,
                                last_active_anomaly_type=anomaly_report.anomaly_type if anomaly_report else None,
                                last_simulated_pnl=pnl_from_previous_l1,
                                system_orchestrator=system_orchestrator,
                                persona_effectiveness_tracker=persona_effectiveness_tracker
                            )
                            logger.info(f"L1_CYCLE_COMPLETED|SimulatedPnL={simulated_pnl:.4f}")
                            # Update for next L1 cycle
                            pnl_from_previous_l1 = simulated_pnl
                        fsm.force_level(0)
                        print("Level 1 cycle complete. Returning to Level 0 monitoring.")
                    if memory_watchdog is not None:
                        memory_watchdog.periodic_check()
                    now = time.time()
                    if now - last_dump_check > state_dump_interval:
                        last_dump_check = now
                        if os.path.exists("DUMP_STATE.trigger"):
                            try:
                                dump_bot_state(fsm, micro_regime_detector, shard_learner)
                                logger.info("STATE_DUMP_TRIGGERED|State dump triggered by DUMP_STATE.trigger file.")
                            except Exception as e:
                                logger.error(f"STATE_DUMP_FAILED|error={e}")
                            try:
                                os.remove("DUMP_STATE.trigger")
                            except Exception:
                                pass
                    time.sleep(fsm.get_current_sleep_interval())
        except KeyboardInterrupt:
            print("Bot shutting down...")
        finally:
            if live_data_manager is not None:
                live_data_manager.stop()
            # Aggressively collect garbage before summary save
            gc.collect()
            try:
                run_summary_gen.generate_and_save_summary()
            except (OSError, MemoryError) as e:
                logger.critical(f"CRITICAL_SHUTDOWN_ERROR|type={type(e).__name__}|msg={e}|Attempting minimal summary save.")
                # Attempt minimal summary save
                try:
                    minimal_summary = {
                        'error_count': getattr(run_summary_gen, 'error_count', -1),
                        'runtime_seconds': time.time() - getattr(run_summary_gen, 'start_time', time.time())
                    }
                    with open("run_summary_minimal.json", "w") as f:
                        json.dump(minimal_summary, f, indent=2)
                except Exception as e2:
                    logger.critical(f"CRITICAL_SHUTDOWN_ERROR|Minimal summary save also failed: {e2}")
            logger.info("[ConfigSelfReflection] Future enhancement: Analyze performance data to suggest config optimizations here.")
    elif args.mode == "L1":
        print("Starting in Level 1 (Manual One-Shot Mode)...")
        latest_price = fsm.get_latest_price
        latest_timestamp = fsm.get_latest_timestamp
        with ErrorContext({'mode': 'L1', 'regime_signal': repr(getattr(fsm, 'latest_regime_signal', None))}):
            run_summary_gen.log_l1_cycle()
            run_level_1_cycle(
                micro_regime_detector,
                fsm,
                memory_watchdog,
                None,
                current_persona,
                current_env_score,
                config,
                run_summary_gen=run_summary_gen,
                shadow_mode_enabled=shadow_mode_enabled,
                shadow_mode_all_adaptive_logic=shadow_mode_enabled,
                opportunistic_actions_live=shadow_mode_enabled,
                anomaly_persistence_tracker=anomaly_persistence_tracker,
                opportunity_action_state=opportunity_action_state,
                opportunity_action_executor=opportunity_action_executor,
                adaptive_penalty_learner=adaptive_penalty_learner,
                last_active_anomaly_type=None,
                last_simulated_pnl=None,
                system_orchestrator=system_orchestrator,
                persona_effectiveness_tracker=persona_effectiveness_tracker
            )
        anomaly_report = shard_learner.detect_anomalies()
        if anomaly_report.is_anomalous:
            logger.warning(f"AnomalyReport: type={anomaly_report.anomaly_type}, strength={anomaly_report.anomaly_strength}, features={anomaly_report.contributing_features}, penalty={anomaly_report.confidence_penalty_suggestion}")
        print("Manual Level 1 cycle complete.")
        if live_data_manager is not None:
            live_data_manager.stop()
            run_summary_gen.generate_and_save_summary()
            logger.info("[ConfigSelfReflection] Future enhancement: Analyze performance data to suggest config optimizations here.")

if __name__ == "__main__":
    main() 