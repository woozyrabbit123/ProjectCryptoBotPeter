import logging
import time
from enum import Enum, auto
from typing import List, Any, Dict, Optional, Tuple, Set, Union, Deque as TypingDeque # Union and Deque were already here
import numpy as np
from collections import deque, defaultdict # deque and defaultdict were already here
# collections is imported below if not already, numpy.lib.stride_tricks is specific
import collections # Keep for other uses like defaultdict if it's not already imported

# Ensure all specified typing imports are present (List, Dict, Optional, Any, Tuple, Set are effectively covered by the first typing import)

# Attempt to import AnomalyReport
try:
    from src.shard_learner import AnomalyReport
except ImportError:
    AnomalyReport = Any # type: ignore

# Attempt to import RegimeSignal
try:
    from src.feature_engineering import RegimeSignal
except ImportError:
    RegimeSignal = Any # type: ignore

# Ensure other specified standard library imports are present
# import time # Already present
# from enum import Enum, auto # Already present
# from collections import deque, defaultdict # Already present

from numpy.lib.stride_tricks import sliding_window_view # This was already present

logger = logging.getLogger(__name__)

class MarketPersona(Enum): 
    """
    Enum representing the bot's market persona (behavioral regime).
    HUNTER: Aggressive, opportunity-seeking.
    GUARDIAN: Neutral, risk-managed.
    GHOST: Defensive, minimal action.
    """
    HUNTER = auto()
    GUARDIAN = auto()
    GHOST = auto()

def calculate_persona(
    recent_anomaly_reports: List[AnomalyReport],
    latency_volatility_index: float,
    regime_stability: float,
    current_env_score: float,
    config: Dict[str, Any],
    persona_effectiveness_tracker: Optional['PersonaEffectivenessTracker'] # Forward reference
) -> MarketPersona:
    """
    Determine the current MarketPersona based on recent anomalies, latency, regime stability, and environmental score.
    Uses config-driven thresholds and environmental modulation.
    Logs persona bias factors if persona_effectiveness_tracker is provided.
    Args:
        recent_anomaly_reports: List of recent anomaly report objects.
        latency_volatility_index: Current latency volatility index.
        regime_stability: Current regime stability metric (0-1).
        current_env_score: Current environmental score (0-1).
        config: Config object with [MarketPersona] section.
        persona_effectiveness_tracker: Optional tracker for logging persona bias factors.
    Returns:
        MarketPersona enum value.
    Side effects:
        Logs threshold calculations and persona bias factors for debugging.
    """
    # Base thresholds from config - minor refactor: assign to local variables
    mp_config = config.get('MarketPersona', {})
    ghost_base_anomaly_pressure: float = mp_config.get('ghost_base_anomaly_pressure', 5.0)
    ghost_base_latency_vol: float = mp_config.get('ghost_base_latency_vol', 3.0)
    hunter_base_anomaly_pressure: float = mp_config.get('hunter_base_anomaly_pressure', 2.0)
    hunter_base_latency_vol: float = mp_config.get('hunter_base_latency_vol', 1.5)
    hunter_base_regime_stability_min: float = mp_config.get('hunter_regime_stability_min', 0.7)
    env_score_modulation_factor: float = mp_config.get('env_score_persona_modulation_factor', 0.6)
    significant_anomaly_strength_threshold: float = mp_config.get('significant_anomaly_strength_threshold', 0.5)
    # Environmental modulation factor
    stability_factor = 1.0 - (current_env_score * env_score_modulation_factor)
    stability_factor = max(0.1, stability_factor)
    # Adjusted thresholds
    adjusted_ghost_anomaly_pressure = ghost_base_anomaly_pressure / stability_factor
    adjusted_ghost_latency_vol = ghost_base_latency_vol / stability_factor
    adjusted_hunter_anomaly_pressure = hunter_base_anomaly_pressure * stability_factor
    adjusted_hunter_latency_vol = hunter_base_latency_vol * stability_factor
    # Calculate anomaly_pressure
    significant_anomalies = [r for r in recent_anomaly_reports if getattr(r, 'is_anomalous', False) and getattr(r, 'anomaly_strength', 0) is not None and r.anomaly_strength > significant_anomaly_strength_threshold]
    anomaly_pressure = sum(r.anomaly_strength for r in significant_anomalies)
    # Persona determination logic
    if (anomaly_pressure > adjusted_ghost_anomaly_pressure or
        latency_volatility_index > adjusted_ghost_latency_vol):
        new_persona = MarketPersona.GHOST
    elif (anomaly_pressure < adjusted_hunter_anomaly_pressure and
          latency_volatility_index < adjusted_hunter_latency_vol and
          regime_stability > hunter_base_regime_stability_min):
        new_persona = MarketPersona.HUNTER
    else:
        new_persona = MarketPersona.GUARDIAN
    # Log the adjusted thresholds used for this calculation for debugging
    logger.debug(f"PersonaCalc: EnvScore={current_env_score:.2f}, StabilityFactor={stability_factor:.2f}, AnomPress={anomaly_pressure:.2f}, LVI={latency_volatility_index:.3f}, RegStab={regime_stability:.2f}")
    logger.debug(f"PersonaCalc Thresholds: GhostAnom={adjusted_ghost_anomaly_pressure:.2f}, GhostLVI={adjusted_ghost_latency_vol:.2f}, HuntAnom={adjusted_hunter_anomaly_pressure:.2f}, HuntLVI={adjusted_hunter_latency_vol:.2f}")
    # Log persona bias factors for each candidate persona
    if persona_effectiveness_tracker is not None:
        for candidate in [MarketPersona.HUNTER, MarketPersona.GUARDIAN, MarketPersona.GHOST]:
            bias = persona_effectiveness_tracker.get_persona_bias_factor(candidate, current_env_score)
            logger.info(f"PERSONA_BIAS_LOG|persona={candidate.name}|env_score={current_env_score:.3f}|bias_factor={bias:.3f}")
    return new_persona

def calculate_env_score(
    anomaly_reports_history: List[AnomalyReport],
    latency_values_history: List[float],
    regime_trend_history: List[float],
    current_persona: MarketPersona,
    config: Dict[str, Any]
) -> float:
    """
    Calculate a 0-1 market stress score from anomaly, latency, and regime trend histories.
    Weighting of components is influenced by the current_persona (from [EnvScoreWeights] config section).
    Args:
        anomaly_reports_history: List of anomaly report objects.
        latency_values_history: List of recent latency values.
        regime_trend_history: List of recent regime trend strengths.
        current_persona: Current MarketPersona.
        config: Config object with [EnvScore] and [EnvScoreWeights] sections.
    Returns:
        Environmental score (float, 0-1).
    """
    env_score_cfg = config.get('EnvScore', {})
    env_score_weights_cfg = config.get('EnvScoreWeights', {})

    def _calculate_anomaly_velocity(reports_hist: List[AnomalyReport], cfg_section: str) -> float: # Added type hints
        cfg_section_params = env_score_cfg 
        lookback: int = cfg_section_params.get('anomaly_velocity_lookback', 20)
        strength_thr: float = cfg_section_params.get('anomaly_velocity_strength_threshold', 1.0)
        norm_factor: float = cfg_section_params.get('anomaly_velocity_max_count_for_norm', 10.0)
        if len(reports_hist) >= lookback:
            recent_sig_anom = [
                r for r in reports_hist[-lookback:] 
                if getattr(r, 'is_anomalous', False) and 
                   getattr(r, 'anomaly_strength', 0) is not None and 
                   r.anomaly_strength > strength_thr
            ]
            return min(len(recent_sig_anom) / norm_factor, 1.0)
        return 0.0

    def _calculate_latency_incoherence(latency_hist: List[float], cfg_section: str) -> float: # Added type hints
        cfg_section_params = env_score_cfg
        lookback: int = cfg_section_params.get('latency_coherence_lookback', 50)
        std_norm_factor: float = cfg_section_params.get('latency_coherence_std_norm_factor', 100.0)
        if len(latency_hist) >= lookback:
            latency_std = np.std(latency_hist[-lookback:])
            coherence = max(0.0, 1.0 - (latency_std / std_norm_factor))
            return 1.0 - coherence
        return 0.5 

    def _calculate_regime_drift(regime_hist: List[float], cfg_section: str) -> float: # Added type hints
        cfg_section_params = env_score_cfg
        lookback: int = cfg_section_params.get('regime_drift_lookback', 20)
        change_thr: float = cfg_section_params.get('regime_drift_change_threshold', 0.1)
        norm_factor: float = cfg_section_params.get('regime_drift_max_count_for_norm', 10.0)
        if len(regime_hist) >= lookback:
            actual_lookback = min(lookback, len(regime_hist) - 1)
            if actual_lookback < 1: return 0.0
            changes = sum(1 for i in range(actual_lookback) if abs(regime_hist[-(i + 1)] - regime_hist[-(i + 2)]) > change_thr)
            return min(changes / norm_factor, 1.0)
        return 0.0

    cfg_section_env_score_name = 'EnvScore' 
    base_components: Dict[str, float] = { # Type hinted
        'anomaly_velocity': _calculate_anomaly_velocity(anomaly_reports_history, cfg_section_env_score_name),
        'latency_incoherence': _calculate_latency_incoherence(latency_values_history, cfg_section_env_score_name),
        'regime_drift': _calculate_regime_drift(regime_trend_history, cfg_section_env_score_name)
    }
    
    persona_str = current_persona.name.lower()
    weights: Dict[str, float] = { # Type hinted
        'anomaly_velocity': env_score_weights_cfg.get(f'{persona_str}_weight_anomaly_velocity', 0.4),
        'latency_incoherence': env_score_weights_cfg.get(f'{persona_str}_weight_latency_incoherence', 0.3),
        'regime_drift': env_score_weights_cfg.get(f'{persona_str}_weight_regime_drift', 0.3)
    }
    total_weight = sum(weights.values())
    if total_weight > 0 and abs(total_weight - 1.0) > 1e-6: # Ensure total_weight is float for abs
        weights = {k: v / total_weight for k, v in weights.items()} # Ensure v / total_weight is float
    
    weights_vec: np.ndarray = np.array([weights[key] for key in weights], dtype=np.float32) # Type hinted
    components_vec: np.ndarray = np.array([base_components[key] for key in weights], dtype=np.float32) # Type hinted
    contextual_env_score: float = float(np.dot(weights_vec, components_vec)) # Type hinted
    logger.debug(f"EnvScoreCalc for Persona {persona_str}: Components={base_components}, Weights={weights}, Score={contextual_env_score:.3f}")
    return min(max(contextual_env_score, 0.0), 1.0)

def evaluate_anomaly_opportunity(
    anomaly_report: AnomalyReport,
    current_persona: MarketPersona,
    env_score: float,
    config: Dict[str, Any],
    **kwargs: Any 
) -> Dict[str, Any]:
    """
    Evaluate if an anomaly presents an opportunity for action. Shadow mode: only logs what would happen.
    """
    opportunity_cfg = config.get('Opportunity', {})
    sharp_type: str = opportunity_cfg.get('sharp_vwma_deviation_anomaly_type', "VwmaDeviationAnomaly")
    
    if (
        getattr(anomaly_report, 'is_anomalous', False) and
        getattr(anomaly_report, 'anomaly_type', None) == sharp_type and
        current_persona == MarketPersona.HUNTER and
        env_score < opportunity_cfg.get('sharp_vwma_max_env_score_for_opp', 0.3)
    ):
        base_regime_confidence: float = kwargs.get('base_regime_confidence', 0.5)
        max_boost_abs: float = opportunity_cfg.get('sharp_vwma_max_confidence_boost_abs', 0.15)
        boost_factor: float = opportunity_cfg.get('sharp_vwma_confidence_boost_factor', 0.3)
        confidence_boost: float = min(max_boost_abs, base_regime_confidence * boost_factor)
        return {
            "opportunity_type": "sharp_vwma_deviation",
            "would_boost_confidence_by": confidence_boost,
            "original_confidence": base_regime_confidence,
            "would_be_new_confidence": base_regime_confidence + confidence_boost,
            "action_taken_shadowed": True
        }
    return {"opportunity_type": "none", "action_taken_shadowed": False}

class AnomalyPersistenceTracker:
    __slots__ = [
        'active_anomalies',
        'logged_durations',
        'persistence_check_interval', 
        'anomaly_timeout',          
        'config',                   
        'anomaly_history'           
    ]
    active_anomalies: Dict[str, Tuple[float, float, float]]
    logged_durations: Dict[str, List[float]] 
    persistence_check_interval: int
    anomaly_timeout: int
    config: Optional[Dict[str, Any]]
    anomaly_history: TypingDeque[Tuple[str, float, float, float]] # Corrected: using TypingDeque

    def __init__(self, persistence_check_interval: int = 30, anomaly_timeout: int = 300, config: Optional[Dict[str, Any]] = None) -> None:
        self.active_anomalies: Dict[str, Tuple[float, float, float]] = {}
        self.logged_durations: Dict[str, List[float]] = {} 
        self.persistence_check_interval: int = persistence_check_interval
        self.anomaly_timeout: int = anomaly_timeout
        self.config: Optional[Dict[str, Any]] = config
        self.anomaly_history: TypingDeque[Tuple[str, float, float, float]] = deque(maxlen=1000)

    def update(self, anomaly_report_obj: Optional[AnomalyReport], current_time: float) -> None:
        if not anomaly_report_obj or not getattr(anomaly_report_obj, 'is_anomalous', False): 
            return
        timed_out_keys: List[str] = []
        for type_str, (start_time, _, last_seen) in self.active_anomalies.items():
            if current_time - last_seen > self.anomaly_timeout:
                duration: float = last_seen - start_time
                logger.info(f"ANOMALY_END_TIMEOUT: Type={type_str}, Duration={duration:.2f}s (timed out)")
                timed_out_keys.append(type_str)
                self.anomaly_history.append((type_str, start_time, last_seen, duration))
        for key in timed_out_keys:
            del self.active_anomalies[key]
        
        if anomaly_report_obj and getattr(anomaly_report_obj, 'is_anomalous', False) and getattr(anomaly_report_obj, 'anomaly_type', None) is not None:
            atype_str: str = anomaly_report_obj.anomaly_type
            strength: float = getattr(anomaly_report_obj, 'anomaly_strength', 0.0) if getattr(anomaly_report_obj, 'anomaly_strength', None) is not None else 0.0
            
            if atype_str not in self.active_anomalies:
                self.active_anomalies[atype_str] = (current_time, strength, current_time)
                logger.info(f"ANOMALY_START: Type={atype_str}, Strength={strength:.2f}")
            else:
                start_time_prev, peak_strength_prev, _ = self.active_anomalies[atype_str]
                new_peak: float = max(peak_strength_prev, strength)
                self.active_anomalies[atype_str] = (start_time_prev, new_peak, current_time)

    def get_active_anomaly_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        current_time: float = time.time()
        for type_str, (start_time, peak_strength, last_seen) in self.active_anomalies.items():
            summary[type_str] = {
                "duration_seconds": current_time - start_time,
                "peak_strength": peak_strength,
                "last_seen_ago_seconds": current_time - last_seen
            }
        return summary
    def get_persistence_score(self, anomaly_type: str) -> float:
        """
        Returns a score [0,1] based on frequency and avg duration of recent anomalies of this type.
        """
        if not anomaly_type:
            return 0.0
        
        ap_config = self.config.get('AnomalyPersistence', {}) if self.config else {}
        lookback: int = ap_config.get('memory_influence_lookback_seconds', 3600)
        freq_max: int = ap_config.get('memory_freq_score_max_count', 5)
        dur_max: int = ap_config.get('memory_duration_score_max_avg_seconds', 300)
            
        now: float = time.time()
        recent: List[Tuple[str, float, float, float]] = [
            h for h in self.anomaly_history if h[0] == anomaly_type and (now - h[2]) < lookback
        ]
        count: int = len(recent)
        avg_duration: float = np.mean([h[3] for h in recent]) if recent else 0.0 # h[3] is duration which is float
        frequency_score: float = min(1.0, count / max(freq_max, 1)) # Avoid division by zero if freq_max is 0
        duration_score: float = min(1.0, avg_duration / max(dur_max, 1)) # Avoid division by zero if dur_max is 0
        return (frequency_score + duration_score) / 2.0

def calculate_memory_influenced_penalty(
    base_penalty: float, 
    anomaly_type: str, 
    anomaly_persistence_tracker: AnomalyPersistenceTracker, 
    config: Dict[str, Any]
) -> float:
    """
    Adjusts base_penalty based on anomaly memory/persistence.
    """
    persistence_score: float = anomaly_persistence_tracker.get_persistence_score(anomaly_type)
    ap_config: Dict[str, Any] = config.get('AnomalyPersistence', {}) # Type hinted ap_config
    high_thr: float = ap_config.get('persistence_high_threshold', 0.7)
    med_thr: float = ap_config.get('persistence_medium_threshold', 0.4)
    mult_high: float = ap_config.get('penalty_multiplier_high_persistence', 1.4)
    mult_med: float = ap_config.get('penalty_multiplier_medium_persistence', 1.2)
    max_penalty: float = ap_config.get('max_penalty_after_memory_influence', 0.8)
    adjusted_penalty: float = base_penalty # Type hinted adjusted_penalty
    if persistence_score > high_thr:
        adjusted_penalty = min(max_penalty, base_penalty * mult_high)
        logger.info(f"MEMORY_INFLUENCE: High persistence ({persistence_score:.2f}) for {anomaly_type}, penalty adjusted to {adjusted_penalty:.3f}")
    elif persistence_score > med_thr:
        adjusted_penalty = min(max_penalty, base_penalty * mult_med)
        logger.info(f"MEMORY_INFLUENCE: Medium persistence ({persistence_score:.2f}) for {anomaly_type}, penalty adjusted to {adjusted_penalty:.3f}")
    return adjusted_penalty

# --- Opportunity Action Logic ---
class OpportunityActionState:
    __slots__ = [
        'action_timestamps',
        'consecutive_actions',
        'last_action_time',        
        'opportunity_active'       
    ]
    action_timestamps: TypingDeque[float] # Using aliased Deque
    consecutive_actions: int
    last_action_time: float
    opportunity_active: Optional[Dict[str, Any]]

    def __init__(self) -> None:
        self.action_timestamps: TypingDeque[float] = deque(maxlen=100) # Initialized with type
        self.consecutive_actions: int = 0
        self.last_action_time: float = 0.0
        self.opportunity_active: Optional[Dict[str, Any]] = None

def execute_opportunity_action(
    opportunity_details_dict: Dict[str, Any], 
    current_regime_signal: Optional[RegimeSignal], 
    config: Dict[str, Any], 
    state_tracker: OpportunityActionState
) -> Optional[float]:
    """
    Executes a live opportunity action if safety limits allow. Returns boosted confidence or None.
    """
    now: float = time.time() # Type hinted
    opp_type: Optional[str] = opportunity_details_dict.get('opportunity_type')
    if opp_type != 'sharp_vwma_deviation':
        return None

    limits_cfg: Dict[str, Any] = config.get('OpportunitySafetyLimits', {}) # Renamed for clarity and typed
    max_per_hour: int = int(limits_cfg.get('max_opportunity_actions_per_hour', 2)) # Explicit int
    max_consecutive: int = int(limits_cfg.get('max_consecutive_opportunity_actions', 1)) # Explicit int
    cooldown: int = int(limits_cfg.get('opportunity_action_cooldown_seconds', 900)) # Explicit int
    min_conf_floor: float = float(limits_cfg.get('opportunity_min_base_confidence_floor', 0.3)) # Explicit float
    
    market_open: bool = limits_cfg.get('opportunity_market_hours_only', 'true').lower() == 'true'
    # This variable is not used further, but kept as per original code.

    state_tracker.action_timestamps = deque([t for t in state_tracker.action_timestamps if now - t < 3600], maxlen=100)
    if len(state_tracker.action_timestamps) >= max_per_hour:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=hourly_limit|count={len(state_tracker.action_timestamps)}|limit={max_per_hour}")
        return None
    if now - state_tracker.last_action_time < cooldown:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=cooldown|since_last={now - state_tracker.last_action_time:.1f}s|cooldown={cooldown}s")
        return None
    if state_tracker.consecutive_actions >= max_consecutive:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=consecutive_limit|count={state_tracker.consecutive_actions}|limit={max_consecutive}")
        return None
    
    base_confidence: float = getattr(current_regime_signal, 'confidence', 0.0) if current_regime_signal else 0.0
    if base_confidence < min_conf_floor:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=confidence_floor|base_confidence={base_confidence:.3f}|floor={min_conf_floor}")
        return None

    opportunity_cfg: Dict[str, Any] = config.get('Opportunity', {}) # Typed
    strength: float = opportunity_details_dict.get('strength', 0.0)
    boost_factor_config: float = opportunity_cfg.get('sharp_vwma_strength_to_boost_factor', 0.3)
    boost_factor: float = min(1.25, 1.0 + (strength * boost_factor_config))
    max_boosted: float = opportunity_cfg.get('sharp_vwma_max_boosted_confidence', 0.95)
    boosted_confidence: float = min(max_boosted, base_confidence * boost_factor) # Type hinted
    logger.info(f"OPPORTUNITY_LIVE|type={opp_type}|base_conf={base_confidence:.3f}|boosted_conf={boosted_confidence:.3f}|factor={boost_factor:.3f}")
    
    state_tracker.action_timestamps.append(now)
    state_tracker.last_action_time = now
    state_tracker.consecutive_actions += 1
    state_tracker.opportunity_active = {
        'original_confidence': base_confidence,
        'boosted_confidence': boosted_confidence,
        'cycles_remaining': 1, # Assuming int
        'id': f"{opp_type}_{int(now)}"
    }
    return boosted_confidence

class PersonaTransitionLimiter:
    min_time_between: int
    max_transitions_in_window: int # Using the renamed attribute from previous successful step
    window_seconds: int
    max_consecutive_blocks_before_force: int # Using the renamed attribute
    last_recorded_transition_time: float # Using the renamed attribute
    transition_timestamps_in_window: List[float] # Using the renamed attribute
    consecutive_blocked_attempts: Dict[Tuple[str, str], int] # Using the renamed attribute

    def __init__(self, config: Dict[str, Any]) -> None:
        mp_config = config.get('MarketPersona', {})
        self.min_time_between: int = mp_config.get('min_seconds_between_persona_transitions', 60)
        self.max_transitions_in_window: int = mp_config.get('max_persona_transitions_in_window', 3) 
        self.window_seconds: int = mp_config.get('persona_transition_check_window_seconds', 300)
        self.max_consecutive_blocks_before_force: int = mp_config.get('max_consecutive_persona_blocks_before_force', 3) 
        self.last_recorded_transition_time: float = 0.0 
        self.transition_timestamps_in_window: List[float] = [] 
        self.consecutive_blocked_attempts: Dict[Tuple[str, str], int] = defaultdict(int)

    def can_transition(self, now: float, current_persona: MarketPersona, intended_persona: MarketPersona) -> bool:
        self.transition_timestamps_in_window = [
            t for t in self.transition_timestamps_in_window if now - t < self.window_seconds
        ]
        
        key: Tuple[str, str] = (current_persona.name, intended_persona.name) # Type hinted local variable

        if self.last_recorded_transition_time > 0 and (now - self.last_recorded_transition_time < self.min_time_between):
            self.consecutive_blocked_attempts[key] += 1
            if self.consecutive_blocked_attempts[key] >= self.max_consecutive_blocks_before_force:
                logger.warning(f"PERSONA_FORCE_TRANSITION|from={current_persona.name}|to={intended_persona.name}|reason=max_consecutive_blocks_met_for_min_time_rule|blocks={self.consecutive_blocked_attempts[key]}")
                return True 
            logger.warning(f"PERSONA_TRANSITION_BLOCKED|reason=too_soon_since_last_global_transition|since_last={now - self.last_recorded_transition_time:.1f}s|min_required={self.min_time_between}s|from={current_persona.name}|to={intended_persona.name}|consecutive_blocks_for_this_pair={self.consecutive_blocked_attempts[key]}")
            return False
        
        if len(self.transition_timestamps_in_window) >= self.max_transitions_in_window:
            self.consecutive_blocked_attempts[key] += 1
            if self.consecutive_blocked_attempts[key] >= self.max_consecutive_blocks_before_force:
                logger.warning(f"PERSONA_FORCE_TRANSITION|from={current_persona.name}|to={intended_persona.name}|reason=max_consecutive_blocks_met_for_window_limit_rule|blocks={self.consecutive_blocked_attempts[key]}")
                return True 
            logger.warning(f"PERSONA_TRANSITION_BLOCKED|reason=too_many_in_window|count={len(self.transition_timestamps_in_window)}|window={self.window_seconds}s|from={current_persona.name}|to={intended_persona.name}|consecutive_blocks_for_this_pair={self.consecutive_blocked_attempts[key]}")
            return False
            
        return True # If not blocked, allow transition. Blocked counter for this pair will be reset by record_transition.

    def record_transition(self, now: float, current_persona: MarketPersona, intended_persona: MarketPersona) -> None:
        self.transition_timestamps_in_window.append(now)
        self.last_recorded_transition_time = now 
        
        key: Tuple[str, str] = (current_persona.name, intended_persona.name) # Type hinted local variable
        if self.consecutive_blocked_attempts[key] > 0: 
            logger.info(f"Resetting consecutive_blocked_attempts for {key} from {self.consecutive_blocked_attempts[key]} to 0 after successful/forced transition.")
        self.consecutive_blocked_attempts[key] = 0 
        
        logger.info(f"PERSONA_TRANSITION_RECORDED|time={now}|from={current_persona.name}|to={intended_persona.name}")

class EnvScoreSmoothing:
    alpha: float
    raw_scores: TypingDeque[float] # Changed from deque[float] to TypingDeque[float]
    smoothed_score: Optional[float]

    def __init__(self, config: Dict[str, Any]) -> None:
        env_score_cfg = config.get('EnvScore', {})
        self.alpha: float = env_score_cfg.get('smoothing_alpha', 0.3)
        self.raw_scores: TypingDeque[float] = deque(maxlen=env_score_cfg.get('smoothing_raw_score_history_length', 10))
        self.smoothed_score: Optional[float] = None

    def update(self, raw_score: float) -> float:
        self.raw_scores.append(raw_score)
        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            self.smoothed_score = self.alpha * raw_score + (1 - self.alpha) * self.smoothed_score
        logger.debug(f"ENV_SCORE_SMOOTHED|raw={raw_score:.3f}|smoothed={self.smoothed_score:.3f}|alpha={self.alpha:.2f}")
        # Ensure a float is always returned, which it is as smoothed_score is float or raw_score (float)
        return self.smoothed_score if self.smoothed_score is not None else raw_score # Should always be float

class AnomalyCircuitBreaker:
    max_anomalies_per_minute: int
    reset_time_seconds: int
    warning_threshold_factor: float
    anomaly_timestamps: List[float]
    circuit_open: bool
    last_state_change: float

    def __init__(self, config: Dict[str, Any]) -> None:
        ad_config = config.get('AnomalyDetection', {})
        self.max_anomalies_per_minute: int = ad_config.get('circuit_breaker_max_anomalies_per_minute', 20)
        self.reset_time_seconds: int = ad_config.get('circuit_breaker_reset_seconds', 300)
        self.warning_threshold_factor: float = ad_config.get('circuit_breaker_warning_threshold_factor', 0.8)
        self.anomaly_timestamps: List[float] = []
        self.circuit_open: bool = False
        self.last_state_change: float = 0.0

    def should_process_anomaly(self, now: float) -> bool:
        self.anomaly_timestamps = [t for t in self.anomaly_timestamps if now - t < 60]
        
        rate: int = len(self.anomaly_timestamps) # Type hinted local variable
        max_rate: int = self.max_anomalies_per_minute # Type hinted local variable
        warning_threshold: float = max_rate * self.warning_threshold_factor # Type hinted local variable

        if not self.circuit_open and rate > warning_threshold and rate < max_rate:
            logger.warning(f"ANOMALY_CIRCUIT_PRE_TRIGGER_WARNING|current_rate={rate:.2f}|threshold={max_rate}")
            
        if self.circuit_open:
            if now - self.last_state_change > self.reset_time_seconds:
                self.circuit_open = False
                self.last_state_change = now
                logger.info(f"ANOMALY_CIRCUIT_CLOSED|time={now}")
                return True 
            else:
                return False 
                
        if rate >= max_rate:
            self.circuit_open = True
            self.last_state_change = now
            logger.warning(f"ANOMALY_CIRCUIT_OPEN|time={now}|count={rate}|limit={max_rate}")
            return False 

        return True 

    def record_anomaly_detected(self, now: float) -> None:
        """Call this only when an actual anomaly is detected (is_anomalous==True)."""
        self.anomaly_timestamps.append(now)

class AnomalySequenceDetector:
    """
    Detects predefined sequences of anomaly types using a fixed-size window.
    Sequences and responses are loaded from config ([AnomalySequences]).
    Uses tuple hashes for efficient matching.
    """
    config: Dict[str, Any]
    seq_window_size: int
    seq_window: TypingDeque[str]
    sequences: List[List[str]]
    seq_hashes: Set[int]
    seq_hash_to_str: Dict[int, str]
    seq_hash_to_response: Dict[int, Dict[str, Any]]
    last_match_hash: Optional[int]

    def __init__(self, main_app_config: Dict[str, Any]) -> None: # Parameter name changed to main_app_config for clarity
        """
        Args:
            main_app_config: The main application config object. 
                             This class will look for an 'AnomalySequences' section within it.
        """
        self.config: Dict[str, Any] = main_app_config 
        
        anomaly_sequences_section_config: Dict[str, Any] = self.config.get('AnomalySequences', {})
        
        self.seq_window_size: int = anomaly_sequences_section_config.get('seq_window_size', 3)
        self.seq_window: TypingDeque[str] = deque(maxlen=self.seq_window_size) # Use TypingDeque
        self.sequences: List[List[str]] = []
        self.seq_hashes: Set[int] = set()
        self.seq_hash_to_str: Dict[int, str] = {}
        self.seq_hash_to_response: Dict[int, Dict[str, Any]] = {}
        
        seqs_str: str = anomaly_sequences_section_config.get('sequences_of_interest', '')
        responses_str: str = anomaly_sequences_section_config.get('suggested_responses', '')

        for seq_item_str in seqs_str.split(';'):
            parts = [s.strip() for s in seq_item_str.split('->') if s.strip()]
            if parts:
                self.sequences.append(parts)
                h = hash(tuple(parts))
                self.seq_hashes.add(h)
                self.seq_hash_to_str[h] = '->'.join(parts)

        for resp_item_str in responses_str.split(';'):
            if ':' in resp_item_str:
                seq_str_part, resp_json_part = resp_item_str.split(':', 1)
                seq_parts_for_resp = [s.strip() for s in seq_str_part.split('->') if s.strip()]
                if seq_parts_for_resp: 
                    h_resp = hash(tuple(seq_parts_for_resp))
                    try:
                        import json 
                        self.seq_hash_to_response[h_resp] = json.loads(resp_json_part)
                    except json.JSONDecodeError: 
                        logger.error(f"Failed to parse JSON response for sequence: '{seq_str_part}' with JSON: '{resp_json_part}'", exc_info=True)
                        self.seq_hash_to_response[h_resp] = {'action': 'log_only', 'error': 'json_decode_failed'}
        self.last_match_hash: Optional[int] = None

    def update(self, new_anomaly_type_str: str) -> Optional[Dict[str, Any]]:
        """
        Add a new anomaly type to the window and check for sequence match.
        Args:
            new_anomaly_type_str: The anomaly type string to append.
        Returns:
            dict with 'sequence' and 'response' if matched, else None.
        Side effects:
            Logs match events.
        """
        self.seq_window.append(new_anomaly_type_str)
        if len(self.seq_window) < self.seq_window_size:
            return None
        current_hash: int = hash(tuple(self.seq_window)) # Type hinted local variable
        if current_hash in self.seq_hashes and current_hash != self.last_match_hash:
            response = self.seq_hash_to_response.get(current_hash, {'action': 'log_only'})
            logger.info(f"ANOMALY_SEQUENCE_MATCH|sequence={self.seq_hash_to_str[current_hash]}|suggested_response_dict={response}")
            self.last_match_hash = current_hash
            return {'sequence': self.seq_hash_to_str[current_hash], 'response': response}
        return None

class OpportunityActionExecutor:
    """
    Handles advanced, vectorized opportunity parameter modifications for live/shadow trading.
    Applies config-driven safety limits and logs all actions.
    """
    config: Dict[str, Any]
    state: Dict[str, float] # Changed Union[float,int] to float as all values are initialized/used as floats.

    def __init__(self, main_app_config: Dict[str, Any]) -> None: 
        """
        Args:
            main_app_config: Main application config object.
        """
        self.config: Dict[str, Any] = main_app_config 
        self.state: Dict[str, float] = {'last_action_time': 0.0, 'actions_this_hour': 0.0, 'last_hour': 0.0} 

    def execute_advanced_opportunity(
        self, 
        opportunity_signal_dict: Dict[str, Any], 
        base_parameters_dict: Dict[str, Any], 
        current_persona_enum: MarketPersona
        # config_dict argument is removed, self.config is used internally.
    ) -> Dict[str, Any]:
        """
        Vectorized parameter modification for advanced opportunity actions.
        Applies safety gates and amplitude clamps from config ([OpportunitySafetyLimits]).
        Args:
            opportunity_signal_dict: Dict describing the opportunity signal.
            base_parameters_dict: Dict of base parameters (size, tp, sl, etc.).
            current_persona_enum: Current MarketPersona.
        Returns:
            Dict describing the action taken or blocked.
        Side effects:
            Logs all actions and warnings for amplitude exceedance.
        """
        now: float = time.time()
        opp_type: Optional[str] = opportunity_signal_dict.get('opportunity_type')
        persona_name: str = current_persona_enum.name # Renamed from persona to persona_name

        limits_cfg: Dict[str,Any] = self.config.get('OpportunitySafetyLimits', {}) # Using self.config
        max_per_hour: int = int(limits_cfg.get('max_opportunity_actions_per_hour', 2))
        min_conf_floor: float = float(limits_cfg.get('opportunity_min_base_confidence_floor', 0.3)) # Renamed min_conf
        max_env_score_allowed: float = float(limits_cfg.get('max_env_score_for_opp', 0.5)) # Renamed max_env_score
        cooldown_seconds: int = int(limits_cfg.get('opportunity_action_cooldown_seconds', 900)) # Renamed cooldown
        max_param_mod_amplitude_factor: float = float(limits_cfg.get('max_param_mod_amplitude_factor', 2.0)) # Renamed max_param_mod_amplitude

        current_hour: int = int(now // 3600) # Renamed hour
        if current_hour != self.state.get('last_hour', 0.0):
            self.state['actions_this_hour'] = 0.0
            self.state['last_hour'] = float(current_hour)

        if self.state.get('actions_this_hour', 0.0) >= max_per_hour:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by hourly limit|type={opp_type}|persona={persona_name}")
            return {'action': 'blocked_hourly_limit'}
        
        if now - self.state.get('last_action_time', 0.0) < cooldown_seconds:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by cooldown|type={opp_type}|persona={persona_name}")
            return {'action': 'blocked_cooldown'}

        base_confidence_val: float = base_parameters_dict.get('base_confidence', 0.0) # Renamed base_conf
        current_env_score_val: float = base_parameters_dict.get('env_score', 0.0) # Renamed env_score

        if base_confidence_val < min_conf_floor:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by confidence floor|base_conf={base_confidence_val:.3f}|floor={min_conf_floor}")
            return {'action': 'blocked_confidence_floor'}
        if current_env_score_val > max_env_score_allowed:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by env_score|env_score={current_env_score_val:.3f}|max={max_env_score_allowed}")
            return {'action': 'blocked_env_score'}

        if opp_type == 'sharp_vwma_deviation' and persona_name == 'HUNTER':
            original_params_np: np.ndarray = np.array([ # Renamed orig
                float(base_parameters_dict.get('size', 1.0)),
                float(base_parameters_dict.get('tp', 1.0)),
                float(base_parameters_dict.get('sl', 1.0))
            ], dtype=np.float32)
            
            opportunity_type_cfg = self.config.get('Opportunity', {}).get(opp_type, {}) 
            
            vwma_deviation_val: float = float(opportunity_signal_dict.get('vwma_deviation', 0.05)) # Renamed vwma_dev
            persona_aggression_factor: float = opportunity_type_cfg.get('persona_aggression_factor', 
                                                                    self.config.get('MarketPersona', {}).get(f'{persona_name}_aggression', 1.1)) # Renamed persona_agg
            env_stability_factor: float = max(0.1, 1.0 - current_env_score_val) # Renamed env_stab
            
            boost_min_clip_val: float = opportunity_type_cfg.get('boost_factor_min_clip', 1.0)
            boost_max_clip_val: float = opportunity_type_cfg.get('boost_factor_max_clip', 1.5)
            
            calculated_boost_factor: float = np.clip( # Renamed boost_factor
                (vwma_deviation_val * persona_aggression_factor) / max(env_stability_factor, 1e-9), 
                boost_min_clip_val, 
                boost_max_clip_val 
            )
            modified_params_np: np.ndarray = original_params_np * calculated_boost_factor # Renamed mods

            param_names: List[str] = ['size', 'tp', 'sl']
            for i, name in enumerate(param_names):
                if (modified_params_np[i] > original_params_np[i] * max_param_mod_amplitude_factor or
                    modified_params_np[i] < original_params_np[i] / max_param_mod_amplitude_factor):
                    logger.warning(f"MAX_PARAM_MOD_AMPLITUDE_EXCEEDED|param={name}|original={original_params_np[i]}|intended={modified_params_np[i]}")
            
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Would modify params (vectorized)|size={modified_params_np[0]:.2f}|tp={modified_params_np[1]:.2f}|sl={modified_params_np[2]:.2f}|type={opp_type}|persona={persona_name}|boost_factor={calculated_boost_factor:.3f}")
            
            self.state['last_action_time'] = now
            self.state['actions_this_hour'] += 1.0
            return {
                'action': 'modify_params',
                'original': {'size': float(original_params_np[0]), 'tp': float(original_params_np[1]), 'sl': float(original_params_np[2])},
                'modified': {'size': float(modified_params_np[0]), 'tp': float(modified_params_np[1]), 'sl': float(modified_params_np[2])},
                'opportunity_type': opp_type,
                'persona': persona_name, 
                'boost_factor': float(calculated_boost_factor) 
            }
        return {'action': 'no_action'}

def detect_hunter_momentum_burst(price_series_np_array: np.ndarray, config: Dict[str, Any]) -> bool:
    """
    Detects if there are 3 consecutive 5-period momentum windows with > X% rise.
    Args:
        price_series_np_array: np.ndarray of recent prices.
        config: Config object with [MarketPersona] hunter_momentum_burst_threshold_pct.
    Returns:
        True if burst detected, else False.
    Side effects:
        Logs detection event.
    """
    if len(price_series_np_array) < 15: 
        return False
    
    mp_config = config.get('MarketPersona', {})
    threshold_pct: float = mp_config.get('hunter_momentum_burst_threshold_pct', 0.03) # Renamed from threshold
    
    if price_series_np_array.ndim == 1: 
        windows: np.ndarray = sliding_window_view(price_series_np_array, window_shape=5)
    else: 
        logger.error("detect_hunter_momentum_burst expects a 1D numpy array for prices.")
        return False
        
    pct_changes: np.ndarray = (windows[:, -1] - windows[:, 0]) / np.maximum(windows[:, 0], 1e-9) 
    above: np.ndarray = pct_changes > threshold_pct # Used renamed threshold_pct
    
    if len(above) < 3:
        return False
    burst_windows: np.ndarray = sliding_window_view(above.astype(int), window_shape=3)
    for i, win in enumerate(burst_windows): # win is also an np.ndarray
        if np.all(win):
            logger.info(f"HUNTER_MOMENTUM_BURST_DETECTED|start_idx={i}|threshold={threshold_pct}")
            return True
    return False

def detect_hunter_vwap_momentum_pattern(
    current_price: float, 
    vwap_or_vwma: float, 
    price_velocity_1min: float, 
    volume_ratio_1min_vs_5min: float, 
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Detects the HUNTER persona's VWAP Momentum Pattern.
    Args:
        current_price: Latest price.
        vwap_or_vwma: Current VWAP or VWMA value.
        price_velocity_1min: 1-minute price change rate.
        volume_ratio_1min_vs_5min: Ratio of 1-min volume to 5-min average volume.
        config: Config object with [HunterPatterns] section.
    Returns:
        dict with pattern details if detected, else None.
    Side effects:
        Logs detection event.
    """
    if vwap_or_vwma == 0: 
        return None
        
    pattern_cfg: Dict[str, Any] = config.get('HunterPatterns', {}) # Renamed hp_config and typed
    vwap_distance_pct: float = (current_price - vwap_or_vwma) / vwap_or_vwma
    
    min_dist: float = pattern_cfg.get('vwap_momentum_min_distance_pct', 0.002) # Renamed
    min_vol_ratio: float = pattern_cfg.get('vwap_momentum_min_volume_ratio', 1.5) # Renamed

    if abs(vwap_distance_pct) > min_dist:
        if np.sign(vwap_distance_pct) == np.sign(price_velocity_1min): 
            if volume_ratio_1min_vs_5min > min_vol_ratio:
                direction: int = int(np.sign(vwap_distance_pct))
                
                max_conf: float = pattern_cfg.get('vwap_max_pattern_confidence', 0.95) # Renamed
                dist_conf_mult: float = pattern_cfg.get('vwap_dist_conf_multiplier', 200.0) 
                vol_conf_div: float = pattern_cfg.get('vwap_vol_conf_divisor', 3.0) 
                
                calculated_confidence: float = min(
                    max_conf, 
                    abs(vwap_distance_pct) * dist_conf_mult * volume_ratio_1min_vs_5min / max(vol_conf_div, 1e-9) 
                )
                logger.info(f"HUNTER_INSTINCT_PATTERN|type=vwap_momentum|direction={direction}|calculated_confidence={calculated_confidence:.3f}")
                return {'pattern_type': 'vwap_momentum', 'direction': direction, 'calculated_confidence': calculated_confidence}
    return None

def apply_guardian_tactical_validation(
    regime_signal_obj: Optional[RegimeSignal], 
    current_env_score: float, 
    recent_anomaly_reports: List[AnomalyReport], 
    config: Dict[str, Any]
) -> float:
    """
    MVP tactical validation for GUARDIAN persona. Returns a multiplier to apply to regime_signal_obj.confidence.
    Args:
        regime_signal_obj: RegimeSignal instance.
        current_env_score: float (smoothed env score).
        recent_anomaly_reports: list of AnomalyReport-like objects.
        config: config object.
    Returns:
        validation_multiplier (float, clamped).
    Side effects:
        Logs each check if triggered.
    """
    validation_multiplier: float = 1.0
    guardian_cfg: Dict[str, Any] = config.get('GuardianTactics', {}) # Renamed gt_config

    high_vol_regime_name: str = guardian_cfg.get('high_vol_regime_name', 'high')
    poor_env_score_threshold: float = guardian_cfg.get('poor_env_score_threshold', 0.6)
    regime_env_mismatch_penalty_factor: float = guardian_cfg.get('regime_env_mismatch_penalty_factor', 0.8)

    if regime_signal_obj and getattr(regime_signal_obj, 'volatility_regime', None) == high_vol_regime_name and \
       current_env_score > poor_env_score_threshold:
        validation_multiplier *= regime_env_mismatch_penalty_factor
        logger.info(f"GUARDIAN_TACTIC|type=regime_env_mismatch|vol_regime={getattr(regime_signal_obj, 'volatility_regime', 'N/A')}|env_score={current_env_score:.2f}|multiplier_applied={regime_env_mismatch_penalty_factor}")

    anomaly_interference_lookback_seconds: float = guardian_cfg.get('anomaly_interference_lookback_seconds', 300.0) 
    anomaly_interference_max_count: int = guardian_cfg.get('anomaly_interference_max_count', 2)
    anomaly_interference_penalty_factor: float = guardian_cfg.get('anomaly_interference_penalty_factor', 0.75)
    
    now: float = time.time()
    significant_anomalies: List[AnomalyReport] = [ # Type hinted
        a for a in recent_anomaly_reports 
        if getattr(a, 'anomaly_strength', 0.0) > 0.5 and 
           (now - getattr(a, 'timestamp', now)) < anomaly_interference_lookback_seconds 
    ]
    if len(significant_anomalies) > anomaly_interference_max_count:
        validation_multiplier *= anomaly_interference_penalty_factor
        logger.info(f"GUARDIAN_TACTIC|type=anomaly_interference|recent_anomaly_count={len(significant_anomalies)}|multiplier_applied={anomaly_interference_penalty_factor}")

    low_rr_min_trend: float = guardian_cfg.get('low_rr_min_trend_for_good_rr', 0.4)
    low_rr_confidence_threshold: float = guardian_cfg.get('low_rr_confidence_threshold', 0.3)
    low_rr_penalty_factor: float = guardian_cfg.get('low_rr_penalty_factor', 0.7)

    trend_strength_val = getattr(regime_signal_obj, 'trend_strength', 0.0) if regime_signal_obj else 0.0
    confidence_val = getattr(regime_signal_obj, 'confidence', 0.0) if regime_signal_obj else 0.0

    if abs(trend_strength_val) < low_rr_min_trend and confidence_val < low_rr_confidence_threshold:
        validation_multiplier *= low_rr_penalty_factor
        logger.info(f"GUARDIAN_TACTIC|type=low_risk_reward|trend={trend_strength_val:.2f}|conf={confidence_val:.2f}|multiplier_applied={low_rr_penalty_factor}")

    min_mult: float = guardian_cfg.get('min_validation_multiplier', 0.5)
    max_mult: float = guardian_cfg.get('max_validation_multiplier', 1.1)
    return float(np.clip(validation_multiplier, min_mult, max_mult))

class AdaptivePenaltyLearner:
    """
    Learns and adapts confidence penalties for each anomaly type based on historical L1 cycle PnL impact.
    Maintains a rolling history of PnL for each anomaly type and adjusts the base penalty accordingly.
    Configurable via [AdaptivePenaltyLearner] section in config.
    """
    config: Dict[str, Any]
    penalty_profiles: defaultdict[str, Dict[str, Union[float, deque[float]]]] 

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.penalty_profiles: defaultdict[str, Dict[str, Union[float, deque[float]]]] = \
            defaultdict(self._default_penalty_profile_factory) 

    def _default_penalty_profile_factory(self) -> Dict[str, Union[float, deque[float]]]: 
        """Factory method to create default penalty profiles."""
        apl_config: Dict[str, Any] = self.config.get('AdaptivePenaltyLearner', {})
        initial_base_penalty: float = apl_config.get('initial_default_base_penalty', 0.15)
        history_len: int = apl_config.get('pnl_impact_history_length', 50)
        
        return { 
            'base_penalty': initial_base_penalty, 
            'pnl_impact_history': deque(maxlen=history_len) # Using collections.deque
        }

    def update_profile_with_l1_outcome(self, anomaly_type: str, simulated_pnl_l1_cycle: float) -> None:
        """
        Update the penalty profile for a given anomaly type with the latest L1 cycle PnL.
        Args:
            anomaly_type: The anomaly type string.
            simulated_pnl_l1_cycle: The PnL (+1/-1/0) of the L1 cycle.
        Side effects:
            May update the base penalty and logs the update.
        """
        if not anomaly_type:
            return
            
        profile = self.penalty_profiles[anomaly_type]
        # Ensure 'pnl_impact_history' is treated as Deque[float]
        pnl_history: TypingDeque[float] = profile['pnl_impact_history'] # type: ignore 
        pnl_history.append(simulated_pnl_l1_cycle)

        apl_config = self.config.get('AdaptivePenaltyLearner', {})
        min_samples_for_adj: int = apl_config.get('min_pnl_samples_for_adjustment', 10)
        
        if len(pnl_history) < min_samples_for_adj:
            return

        recent_impact_lookback: int = apl_config.get('recent_pnl_impact_lookback', 10)
        recent_pnl_list: List[float] = list(pnl_history)[-recent_impact_lookback:]
        
        if not recent_pnl_list: # Should not happen if len(pnl_history) >= min_samples_for_adj (which is >=10 typically)
            return
            
        avg_recent_impact: float = np.mean(recent_pnl_list)
        old_penalty: float = profile['base_penalty'] # type: ignore

        benign_impact_thresh: float = apl_config.get('benign_pnl_impact_threshold', -0.01)
        dangerous_impact_thresh: float = apl_config.get('dangerous_pnl_impact_threshold', -0.05)
        penalty_reduct_factor: float = apl_config.get('penalty_reduction_factor', 0.95)
        penalty_incr_factor: float = apl_config.get('penalty_increase_factor', 1.10)
        min_penalty_cap_val: float = apl_config.get('min_penalty_cap', 0.01)
        max_penalty_cap_val: float = apl_config.get('max_penalty_cap', 0.5)
        if avg_recent_impact > benign_impact_threshold:
            profile['base_penalty'] *= penalty_reduction_factor
        elif avg_recent_impact < dangerous_impact_threshold:
            profile['base_penalty'] *= penalty_increase_factor
        
        current_base_penalty = profile['base_penalty'] # type: ignore
        if avg_recent_impact > benign_impact_thresh:
            current_base_penalty *= penalty_reduct_factor
        elif avg_recent_impact < dangerous_impact_thresh:
            current_base_penalty *= penalty_incr_factor
        
        profile['base_penalty'] = float(np.clip(current_base_penalty, min_penalty_cap_val, max_penalty_cap_val))

        if abs(old_penalty - profile['base_penalty']) > 1e-4: # type: ignore
            logger.info(f"ADAPTIVE_PENALTY_UPDATE|anomaly_type={anomaly_type}|old_penalty={old_penalty:.3f}|new_penalty={profile['base_penalty']:.3f}|avg_recent_pnl={avg_recent_impact:.3f}|pnl_history_len={len(pnl_history)}")

    def get_learned_penalty(self, anomaly_type: str) -> float:
        """
        Returns the learned base_penalty for the anomaly_type.
        If never seen, returns the initial default penalty.
        """
        # 'base_penalty' is float
        return self.penalty_profiles[anomaly_type]['base_penalty'] # type: ignore

class PersonaEffectivenessTracker:
    """
    Tracks persona effectiveness (simulated win rate) by environmental context and provides a bias factor for persona selection.
    Uses a Bayesian-inspired update for success rate and calculates a bias factor for each persona/env context.
    Configurable via [PersonaEffectiveness] section in config.
    """
    config: Dict[str, Any]
    # Internal type alias for the value of the innermost dictionary
    _ContextProfileItemValue = Union[TypingDeque[int], float, int]
    # Internal type alias for the innermost dictionary itself
    _ContextProfileType = Dict[str, _ContextProfileItemValue]
    # Internal type alias for the middle dictionary (env_context_str -> _ContextProfileType), created by defaultdict
    _EnvContextMapType = defaultdict[str, _ContextProfileType]
    
    # Type for the outermost dictionary (persona_name_str -> _EnvContextMapType), created by defaultdict
    persona_outcomes_by_context: defaultdict[str, _EnvContextMapType]
    
    env_score_bins: List[float]
    min_obs_for_bias: int
    bias_strength_factor: float

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pe_config = self.config.get('PersonaEffectiveness', {})
        
        # The lambda for the outer defaultdict returns an _EnvContextMapType,
        # which itself is a defaultdict that uses _default_context_profile_factory.
        self.persona_outcomes_by_context = defaultdict(lambda: defaultdict(self._default_context_profile_factory))
        
        self.env_score_bins = [
            pe_config.get('env_score_bin_low_medium_threshold', 0.3),
            pe_config.get('env_score_bin_medium_high_threshold', 0.6)
        ]
        self.min_obs_for_bias: int = pe_config.get('min_observations_for_bias', 10)
        self.bias_strength_factor: float = pe_config.get('persona_bias_strength_factor', 0.05)

    def _default_context_profile_factory(self) -> _ContextProfileType:
        """Factory method to create default context profiles for the inner defaultdict."""
        # Ensure pe_config is accessed safely within the factory
        pe_config = self.config.get('PersonaEffectiveness', {}) 
        history_len: int = pe_config.get('outcome_history_length_per_context', 30)
        # This dictionary is a _ContextProfileType
        return { 
            'outcomes': deque(maxlen=history_len), # TypingDeque[int] matches _ContextProfileItemValue
            'success_rate_estimate': 0.5,          # float matches _ContextProfileItemValue
            'total_observations': 0                # int matches _ContextProfileItemValue
        }

    def _get_env_context_str(self, env_score: float) -> str:
        if env_score < self.env_score_bins[0]: return "low_env_score"
        if env_score < self.env_score_bins[1]: return "medium_env_score"
        return "high_env_score"

    def record_l1_outcome(self, persona: MarketPersona, env_score: float, simulated_pnl: float) -> None:
        """
        Record the outcome of an L1 cycle for a persona in a given env context.
        Args:
            persona: MarketPersona enum.
            env_score: Environmental score at the time.
            simulated_pnl: Simulated PnL (+1/-1/0).
        Side effects:
            Updates internal stats and logs update.
        """
        persona_name: str = persona.name
        env_context_str: str = self._get_env_context_str(env_score)
        profile = self.persona_outcomes_by_context[persona_name][env_context_str]
        
        outcomes_deque: TypingDeque[int] = profile['outcomes'] # type: ignore
        outcomes_deque.append(1 if simulated_pnl > 0 else 0)
        
        profile['total_observations'] = profile['total_observations'] + 1 # type: ignore
        
        # Bayesian update with uniform prior Beta(1,1)
        successes: int = sum(outcomes_deque)
        total_samples: int = len(outcomes_deque)
        profile['success_rate_estimate'] = (1 + successes) / (2 + total_samples) # type: ignore
        
        logger.debug(f"PERSONA_EFFECTIVENESS_UPDATE|persona={persona_name}|env_context={env_context_str}|pnl={simulated_pnl:.2f}|new_success_rate_est={profile['success_rate_estimate']:.3f}|obs={profile['total_observations']}")

    def get_persona_bias_factor(self, persona_to_evaluate: MarketPersona, current_env_score: float) -> float:
        """
        Returns a bias factor (e.g., 0.95 to 1.05) for a given persona in the current env context.
        Factor > 1.0 means favor this persona, < 1.0 means disfavor.
        Args:
            persona_to_evaluate: MarketPersona enum.
            current_env_score: Current environmental score.
        Returns:
            Bias factor (float).
        Side effects:
            Logs bias calculation if non-neutral.
        """
        persona_name_eval: str = persona_to_evaluate.name
        env_context_str_eval: str = self._get_env_context_str(current_env_score)
        profile_eval = self.persona_outcomes_by_context[persona_name_eval][env_context_str_eval]

        if profile_eval['total_observations'] < self.min_obs_for_bias: # type: ignore
            return 1.0
        
        # Ensure types for calculation
        success_rate_est: float = profile_eval['success_rate_estimate'] # type: ignore
        effectiveness_delta: float = success_rate_est - 0.5 
        bias_val: float = 1.0 + (effectiveness_delta * 2 * self.bias_strength_factor)

        pe_config = self.config.get('PersonaEffectiveness', {})
        min_bias_cfg: float = pe_config.get('min_persona_bias_factor', 0.9)
        max_bias_cfg: float = pe_config.get('max_persona_bias_factor', 1.1)
        
        clamped_bias_val = float(np.clip(bias_val, min_bias_cfg, max_bias_cfg))
        
        if abs(clamped_bias_val - 1.0) > 1e-4:
            logger.info(f"PERSONA_BIAS_CALCULATED|persona={persona_name_eval}|env_context={env_context_str_eval}|success_rate_est={success_rate_est:.3f}|bias_factor={clamped_bias_val:.3f}")
        return clamped_bias_val

# === MVP Layered Decision Architecture & Ensemble Confidence Detector ===
class InstinctiveLayer:
    config: Dict[str, Any]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config

    def evaluate(self, market_state_snapshot: Dict[str, Any], persona_context_dict: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("InstinctiveLayer.evaluate called | persona=%s", persona_context_dict.get('persona'))
        return {'action_bias': None, 'confidence_mod': 0.0}

class TacticalLayer:
    config: Dict[str, Any]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config

    def evaluate(self, market_state_snapshot: Dict[str, Any], persona_context_dict: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("TacticalLayer.evaluate called | persona=%s", persona_context_dict.get('persona'))
        return {'action_bias': None, 'confidence_mod': 0.0}

class StrategicLayer:
    config: Dict[str, Any]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config

    def evaluate(self, market_state_snapshot: Dict[str, Any], persona_context_dict: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("StrategicLayer.evaluate called | persona=%s", persona_context_dict.get('persona'))
        return {'action_bias': None, 'confidence_mod': 0.0}

class LayeredDecisionArchitecture:
    config: Dict[str, Any]
    instinctive: InstinctiveLayer
    tactical: TacticalLayer
    strategic: StrategicLayer

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.instinctive: InstinctiveLayer = InstinctiveLayer(config)
        self.tactical: TacticalLayer = TacticalLayer(config)
        self.strategic: StrategicLayer = StrategicLayer(config)

    def get_layer_weights(self, persona: MarketPersona) -> Dict[str, float]:
        persona_str: str = persona.name.lower() # Renamed from persona_str_lower
        section: str = 'DecisionLayers' # Added type hint for section
        dl_config: Dict[str, Any] = self.config.get(section, {}) # Added type hint and safer get
        return {
            'instinctive': dl_config.get(f'{persona_str}_weight_instinctive', 0.33),
            'tactical': dl_config.get(f'{persona_str}_weight_tactical', 0.33),
            'strategic': dl_config.get(f'{persona_str}_weight_strategic', 0.34),
        }

    def process_l1_decision(
        self, 
        market_state_snapshot: Dict[str, Any], 
        active_persona: MarketPersona, 
        base_regime_signal: Optional[RegimeSignal], 
        shadow_mode: bool = True
    ) -> Optional[RegimeSignal]:
        if base_regime_signal is None: 
            logger.warning("LayeredDecisionArchitecture.process_l1_decision called with None base_regime_signal.")
            return None

        weights: Dict[str, float] = self.get_layer_weights(active_persona)
        persona_ctx: Dict[str, str] = {'persona': active_persona.name}
        
        # The return type of evaluate is Dict[str, Any], so decisions matches this.
        decisions: Dict[str, Dict[str, Any]] = {
            'instinctive': self.instinctive.evaluate(market_state_snapshot, persona_ctx),
            'tactical': self.tactical.evaluate(market_state_snapshot, persona_ctx),
            'strategic': self.strategic.evaluate(market_state_snapshot, persona_ctx),
        }
        logger.info(f"LAYERED_DECISION_ARCH|persona={active_persona.name}|weights={weights}|decisions={decisions}")
        
        avg_conf_mod: float = np.mean([d.get('confidence_mod', 0.0) for d in decisions.values()]) # np.mean returns np.float64
        
        current_confidence: float = getattr(base_regime_signal, 'confidence', 0.5) 
        new_confidence: float = 0.0 # Initialize for both branches

        if not shadow_mode:
            new_confidence = float(np.clip(current_confidence + avg_conf_mod, 0.0, 1.0))
            try:
                base_regime_signal.confidence = new_confidence # Assumes mutable RegimeSignal
                logger.info(f"LAYERED_DECISION_APPLIED|old_conf={current_confidence:.3f}|avg_conf_mod={avg_conf_mod:.3f}|new_conf={new_confidence:.3f}")
            except AttributeError: 
                 logger.error("Cannot directly modify confidence on base_regime_signal. It might be immutable.")
                 return base_regime_signal
        else:
            would_be_conf: float = float(np.clip(current_confidence + avg_conf_mod, 0.0, 1.0))
            logger.info(f"LAYERED_DECISION_SHADOW|avg_conf_mod={avg_conf_mod:.3f}|current_conf={current_confidence:.3f}|would_be_conf={would_be_conf:.3f}")
            
        return base_regime_signal

class EnsembleConfidenceDetector:
    config: Dict[str, Any]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config

    def detect_super_confidence_conditions(self, system_state_snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current_persona_snap: Optional[MarketPersona] = system_state_snapshot.get('current_persona')
        regime_signal_snap: Optional[RegimeSignal] = system_state_snapshot.get('regime_signal')
        env_score_snap: Optional[float] = system_state_snapshot.get('env_score')

        if current_persona_snap and current_persona_snap.name == 'HUNTER':
            ec_config = self.config.get('EnsembleConfidence', {})
            min_conviction_cfg: float = ec_config.get('hunter_confluence_regime_conviction_min', 0.7)
            max_env_stress_cfg: float = ec_config.get('hunter_confluence_max_env_stress', 0.4)
            boost_multiplier_cfg: float = ec_config.get('hunter_confluence_boost_multiplier', 1.2)
            boost_duration_cycles_cfg: int = ec_config.get('hunter_confluence_boost_duration_l1_cycles', 3)

            regime_confidence = getattr(regime_signal_snap, 'confidence', 0.0) if regime_signal_snap else 0.0

            if regime_signal_snap and regime_confidence > min_conviction_cfg and \
               env_score_snap is not None and env_score_snap < max_env_stress_cfg:
                logger.info(f"ENSEMBLE_CONFIDENCE_DETECTED|type=hunter_confluence|multiplier={boost_multiplier_cfg}|duration={boost_duration_cycles_cfg}")
                return {
                    'boost_type': 'hunter_confluence', 
                    'confidence_multiplier': boost_multiplier_cfg, 
                    'duration_cycles': boost_duration_cycles_cfg
                }
        return None