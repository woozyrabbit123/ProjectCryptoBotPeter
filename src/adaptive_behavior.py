import logging
import time
from enum import Enum, auto
from typing import List, Any, Dict
import numpy as np
from collections import deque
import collections
from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict

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
    recent_anomaly_reports: List[Any],
    latency_volatility_index: float,
    regime_stability: float,
    current_env_score: float,
    config: Any,
    persona_effectiveness_tracker: Any = None
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
    # Base thresholds from config
    ghost_base_anomaly_pressure = config.getfloat('MarketPersona', 'ghost_base_anomaly_pressure', fallback=5.0)
    ghost_base_latency_vol = config.getfloat('MarketPersona', 'ghost_base_latency_vol', fallback=3.0)
    hunter_base_anomaly_pressure = config.getfloat('MarketPersona', 'hunter_base_anomaly_pressure', fallback=2.0)
    hunter_base_latency_vol = config.getfloat('MarketPersona', 'hunter_base_latency_vol', fallback=1.5)
    hunter_base_regime_stability_min = config.getfloat('MarketPersona', 'hunter_regime_stability_min', fallback=0.7)
    env_score_modulation_factor = config.getfloat('MarketPersona', 'env_score_persona_modulation_factor', fallback=0.6)
    significant_anomaly_strength_threshold = config.getfloat('MarketPersona', 'significant_anomaly_strength_threshold', fallback=0.5)
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
    anomaly_reports_history: List[Any],
    latency_values_history: List[float],
    regime_trend_history: List[float],
    current_persona: MarketPersona,
    config: Any
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
    def _calculate_anomaly_velocity(reports_hist, cfg_section):
        lookback = config.getint(cfg_section, 'anomaly_velocity_lookback', fallback=20)
        strength_thr = config.getfloat(cfg_section, 'anomaly_velocity_strength_threshold', fallback=1.0)
        norm_factor = config.getfloat(cfg_section, 'anomaly_velocity_max_count_for_norm', fallback=10.0)
        if len(reports_hist) >= lookback:
            recent_sig_anom = [r for r in reports_hist[-lookback:] if getattr(r, 'is_anomalous', False) and getattr(r, 'anomaly_strength', 0) is not None and r.anomaly_strength > strength_thr]
            return min(len(recent_sig_anom) / norm_factor, 1.0)
        return 0.0
    def _calculate_latency_incoherence(latency_hist, cfg_section):
        lookback = config.getint(cfg_section, 'latency_coherence_lookback', fallback=50)
        std_norm_factor = config.getfloat(cfg_section, 'latency_coherence_std_norm_factor', fallback=100.0)
        if len(latency_hist) >= lookback:
            latency_std = np.std(latency_hist[-lookback:])
            coherence = max(0.0, 1.0 - (latency_std / std_norm_factor))
            return 1.0 - coherence
        return 0.5
    def _calculate_regime_drift(regime_hist, cfg_section):
        lookback = config.getint(cfg_section, 'regime_drift_lookback', fallback=20)
        change_thr = config.getfloat(cfg_section, 'regime_drift_change_threshold', fallback=0.1)
        norm_factor = config.getfloat(cfg_section, 'regime_drift_max_count_for_norm', fallback=10.0)
        if len(regime_hist) >= lookback:
            actual_lookback = min(lookback, len(regime_hist) -1)
            if actual_lookback < 1: return 0.0
            changes = sum(1 for i in range(actual_lookback) if abs(regime_hist[-(i+1)] - regime_hist[-(i+2)]) > change_thr)
            return min(changes / norm_factor, 1.0)
        return 0.0
    cfg_section_env_score = 'EnvScore'
    base_components = {
        'anomaly_velocity': _calculate_anomaly_velocity(anomaly_reports_history, cfg_section_env_score),
        'latency_incoherence': _calculate_latency_incoherence(latency_values_history, cfg_section_env_score),
        'regime_drift': _calculate_regime_drift(regime_trend_history, cfg_section_env_score)
    }
    persona_str = current_persona.name
    weights = {
        'anomaly_velocity': config.getfloat('EnvScoreWeights', f'{persona_str.lower()}_weight_anomaly_velocity', fallback=0.4),
        'latency_incoherence': config.getfloat('EnvScoreWeights', f'{persona_str.lower()}_weight_latency_incoherence', fallback=0.3),
        'regime_drift': config.getfloat('EnvScoreWeights', f'{persona_str.lower()}_weight_regime_drift', fallback=0.3)
    }
    total_weight = sum(weights.values())
    if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
        weights = {k: v / total_weight for k, v in weights.items()}
    # Vectorized calculation
    weights_vec = np.array([weights[key] for key in weights], dtype=np.float32)
    components_vec = np.array([base_components[key] for key in weights], dtype=np.float32)
    contextual_env_score = float(np.dot(weights_vec, components_vec))
    logger.debug(f"EnvScoreCalc for Persona {persona_str}: Components={base_components}, Weights={weights}, Score={contextual_env_score:.3f}")
    return min(max(contextual_env_score, 0.0), 1.0)

def evaluate_anomaly_opportunity(
    anomaly_report: Any,
    current_persona: MarketPersona,
    env_score: float,
    config: Any,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate if an anomaly presents an opportunity for action. Shadow mode: only logs what would happen.
    """
    # --- SHADOW MODE: sharp VWMA deviation opportunity ---
    sharp_type = config.get('Opportunity', 'sharp_vwma_deviation_anomaly_type', fallback="VwmaDeviationAnomaly")
    if (
        getattr(anomaly_report, 'is_anomalous', False) and
        getattr(anomaly_report, 'anomaly_type', None) == sharp_type and
        current_persona == MarketPersona.HUNTER and
        env_score < config.getfloat('Opportunity', 'sharp_vwma_max_env_score_for_opp', fallback=0.3)
    ):
        base_regime_confidence = kwargs.get('base_regime_confidence', 0.5)
        max_boost_abs = config.getfloat('Opportunity', 'sharp_vwma_max_confidence_boost_abs', fallback=0.15)
        boost_factor = config.getfloat('Opportunity', 'sharp_vwma_confidence_boost_factor', fallback=0.3)
        confidence_boost = min(max_boost_abs, base_regime_confidence * boost_factor)
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
    def __init__(self, persistence_check_interval: int = 30, anomaly_timeout: int = 300, config=None):
        self.active_anomalies = {}  # {type_str: (start_time_float, peak_strength_float, last_seen_time_float)}
        self.logged_durations = {} # {type_str: [durations_list]}
        self.persistence_check_interval = persistence_check_interval
        self.anomaly_timeout = anomaly_timeout
        self.config = config
        self.anomaly_history = deque(maxlen=1000)  # Store (type, start, end, duration)
    def update(self, anomaly_report_obj: Any, current_time: float):
        if not anomaly_report_obj or not hasattr(anomaly_report_obj, 'is_anomalous'):
            return
        timed_out_keys = []
        for type_str, (start_time, _, last_seen) in self.active_anomalies.items():
            if current_time - last_seen > self.anomaly_timeout:
                duration = last_seen - start_time
                logger.info(f"ANOMALY_END_TIMEOUT: Type={type_str}, Duration={duration:.2f}s (timed out)")
                timed_out_keys.append(type_str)
                # Log to history
                self.anomaly_history.append((type_str, start_time, last_seen, duration))
        for key in timed_out_keys:
            del self.active_anomalies[key]
        if anomaly_report_obj.is_anomalous and anomaly_report_obj.anomaly_type is not None:
            atype_str = anomaly_report_obj.anomaly_type
            strength = anomaly_report_obj.anomaly_strength if anomaly_report_obj.anomaly_strength is not None else 0.0
            if atype_str not in self.active_anomalies:
                self.active_anomalies[atype_str] = (current_time, strength, current_time)
                logger.info(f"ANOMALY_START: Type={atype_str}, Strength={strength:.2f}")
            else:
                start_time, peak_strength, _ = self.active_anomalies[atype_str]
                new_peak = max(peak_strength, strength)
                self.active_anomalies[atype_str] = (start_time, new_peak, current_time)
    def get_active_anomaly_summary(self) -> dict:
        summary = {}
        current_time = time.time()
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
        lookback = (self.config.getint('AnomalyPersistence', 'memory_influence_lookback_seconds', fallback=3600)
                    if self.config else 3600)
        freq_max = (self.config.getint('AnomalyPersistence', 'memory_freq_score_max_count', fallback=5)
                    if self.config else 5)
        dur_max = (self.config.getint('AnomalyPersistence', 'memory_duration_score_max_avg_seconds', fallback=300)
                   if self.config else 300)
        now = time.time()
        recent = [h for h in self.anomaly_history if h[0] == anomaly_type and (now - h[2]) < lookback]
        count = len(recent)
        avg_duration = np.mean([h[3] for h in recent]) if recent else 0.0
        frequency_score = min(1.0, count / freq_max)
        duration_score = min(1.0, avg_duration / dur_max)
        return (frequency_score + duration_score) / 2.0

def calculate_memory_influenced_penalty(base_penalty: float, anomaly_type: str, anomaly_persistence_tracker: AnomalyPersistenceTracker, config: dict) -> float:
    """
    Adjusts base_penalty based on anomaly memory/persistence.
    """
    persistence_score = anomaly_persistence_tracker.get_persistence_score(anomaly_type)
    high_thr = config.getfloat('AnomalyPersistence', 'persistence_high_threshold', fallback=0.7)
    med_thr = config.getfloat('AnomalyPersistence', 'persistence_medium_threshold', fallback=0.4)
    mult_high = config.getfloat('AnomalyPersistence', 'penalty_multiplier_high_persistence', fallback=1.4)
    mult_med = config.getfloat('AnomalyPersistence', 'penalty_multiplier_medium_persistence', fallback=1.2)
    max_penalty = config.getfloat('AnomalyPersistence', 'max_penalty_after_memory_influence', fallback=0.8)
    adjusted_penalty = base_penalty
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
    def __init__(self):
        self.action_timestamps = deque(maxlen=100)
        self.consecutive_actions = 0
        self.last_action_time = 0.0
        self.opportunity_active = None  # dict with keys: original_confidence, boosted_confidence, cycles_remaining, id

def execute_opportunity_action(opportunity_details_dict, current_regime_signal, config, state_tracker: OpportunityActionState):
    """
    Executes a live opportunity action if safety limits allow. Returns boosted confidence or None.
    """
    now = time.time()
    opp_type = opportunity_details_dict.get('opportunity_type')
    if opp_type != 'sharp_vwma_deviation':
        return None
    # Safety limits
    limits = config['OpportunitySafetyLimits'] if 'OpportunitySafetyLimits' in config else {}
    max_per_hour = int(limits.get('max_opportunity_actions_per_hour', 2))
    max_consecutive = int(limits.get('max_consecutive_opportunity_actions', 1))
    cooldown = int(limits.get('opportunity_action_cooldown_seconds', 900))
    min_conf_floor = float(limits.get('opportunity_min_base_confidence_floor', 0.3))
    # Market hours check (for crypto, always true, but placeholder)
    if limits.get('opportunity_market_hours_only', 'true').lower() == 'true':
        market_open = True
    else:
        market_open = True
    # Enforce per-hour and cooldown
    state_tracker.action_timestamps = deque([t for t in state_tracker.action_timestamps if now - t < 3600], maxlen=100)
    if len(state_tracker.action_timestamps) >= max_per_hour:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=hourly_limit|count={len(state_tracker.action_timestamps)}|limit={max_per_hour}")
        return None
    if now - state_tracker.last_action_time < cooldown:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=cooldown|since_last={now-state_tracker.last_action_time:.1f}s|cooldown={cooldown}s")
        return None
    if state_tracker.consecutive_actions >= max_consecutive:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=consecutive_limit|count={state_tracker.consecutive_actions}|limit={max_consecutive}")
        return None
    base_confidence = getattr(current_regime_signal, 'confidence', 0.0)
    if base_confidence < min_conf_floor:
        logger.info(f"OPPORTUNITY_BLOCKED|reason=confidence_floor|base_confidence={base_confidence:.3f}|floor={min_conf_floor}")
        return None
    # Calculate boost
    strength = opportunity_details_dict.get('strength', 0.0)
    boost_factor = min(1.25, 1.0 + (strength * config.getfloat('Opportunity', 'sharp_vwma_strength_to_boost_factor', fallback=0.3)))
    max_boosted = config.getfloat('Opportunity', 'sharp_vwma_max_boosted_confidence', fallback=0.95)
    boosted_confidence = min(max_boosted, base_confidence * boost_factor)
    logger.info(f"OPPORTUNITY_LIVE|type={opp_type}|base_conf={base_confidence:.3f}|boosted_conf={boosted_confidence:.3f}|factor={boost_factor:.3f}")
    # Track state
    state_tracker.action_timestamps.append(now)
    state_tracker.last_action_time = now
    state_tracker.consecutive_actions += 1
    state_tracker.opportunity_active = {
        'original_confidence': base_confidence,
        'boosted_confidence': boosted_confidence,
        'cycles_remaining': 1,
        'id': f"{opp_type}_{int(now)}"
    }
    return boosted_confidence

class PersonaTransitionLimiter:
    def __init__(self, config):
        self.min_time_between = config.getint('MarketPersona', 'min_seconds_between_persona_transitions', fallback=60)
        self.max_transitions = config.getint('MarketPersona', 'max_persona_transitions_in_window', fallback=3)
        self.window_seconds = config.getint('MarketPersona', 'persona_transition_check_window_seconds', fallback=300)
        self.max_consecutive_blocks = config.getint('MarketPersona', 'max_consecutive_persona_blocks_before_force', fallback=3)
        self.last_transition_time = 0.0
        self.transition_times = []  # list of timestamps
        self.consecutive_blocks = {}  # (from, to) -> count
    def can_transition(self, now, current_persona, intended_persona):
        # Remove old transitions
        self.transition_times = [t for t in self.transition_times if now - t < self.window_seconds]
        key = (current_persona.name, intended_persona.name)
        if self.transition_times and (now - self.transition_times[-1] < self.min_time_between):
            self.consecutive_blocks[key] = self.consecutive_blocks.get(key, 0) + 1
            if self.consecutive_blocks[key] > self.max_consecutive_blocks:
                logger.warning(f"PERSONA_FORCE_TRANSITION|from={current_persona.name}|to={intended_persona.name}|blocks_exceeded={self.consecutive_blocks[key]}")
                self.consecutive_blocks[key] = 0
                return True
            logger.warning(f"PERSONA_TRANSITION_BLOCKED|reason=too_soon|since_last={now-self.transition_times[-1]:.1f}s|min_required={self.min_time_between}s|from={current_persona.name}|to={intended_persona.name}|consecutive_blocks={self.consecutive_blocks[key]}")
            return False
        if len(self.transition_times) >= self.max_transitions:
            self.consecutive_blocks[key] = self.consecutive_blocks.get(key, 0) + 1
            if self.consecutive_blocks[key] > self.max_consecutive_blocks:
                logger.warning(f"PERSONA_FORCE_TRANSITION|from={current_persona.name}|to={intended_persona.name}|blocks_exceeded={self.consecutive_blocks[key]}")
                self.consecutive_blocks[key] = 0
                return True
            logger.warning(f"PERSONA_TRANSITION_BLOCKED|reason=too_many|count={len(self.transition_times)}|window={self.window_seconds}s|from={current_persona.name}|to={intended_persona.name}|consecutive_blocks={self.consecutive_blocks[key]}")
            return False
        self.consecutive_blocks[key] = 0
        return True
    def record_transition(self, now, current_persona, intended_persona):
        self.transition_times.append(now)
        key = (current_persona.name, intended_persona.name)
        self.consecutive_blocks[key] = 0
        logger.info(f"PERSONA_TRANSITION_RECORDED|time={now}|from={current_persona.name}|to={intended_persona.name}")

class EnvScoreSmoothing:
    def __init__(self, config):
        self.alpha = config.getfloat('EnvScore', 'smoothing_alpha', fallback=0.3)
        self.raw_scores = deque(maxlen=config.getint('EnvScore', 'smoothing_raw_score_history_length', fallback=10))
        self.smoothed_score = None
    def update(self, raw_score):
        self.raw_scores.append(raw_score)
        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            self.smoothed_score = self.alpha * raw_score + (1 - self.alpha) * self.smoothed_score
        logger.debug(f"ENV_SCORE_SMOOTHED|raw={raw_score:.3f}|smoothed={self.smoothed_score:.3f}|alpha={self.alpha:.2f}")
        return self.smoothed_score

class AnomalyCircuitBreaker:
    def __init__(self, config):
        self.max_anomalies_per_minute = config.getint('AnomalyDetection', 'circuit_breaker_max_anomalies_per_minute', fallback=20)
        self.reset_time_seconds = config.getint('AnomalyDetection', 'circuit_breaker_reset_seconds', fallback=300)
        self.warning_threshold_factor = config.getfloat('AnomalyDetection', 'circuit_breaker_warning_threshold_factor', fallback=0.8)
        self.anomaly_timestamps = []
        self.circuit_open = False
        self.last_state_change = 0.0
    def should_process_anomaly(self, now):
        # Remove old timestamps
        self.anomaly_timestamps = [t for t in self.anomaly_timestamps if now - t < 60]
        rate = len(self.anomaly_timestamps)
        max_rate = self.max_anomalies_per_minute
        warning_threshold = max_rate * self.warning_threshold_factor
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
    def record_anomaly_detected(self, now):
        """Call this only when an actual anomaly is detected (is_anomalous==True)."""
        self.anomaly_timestamps.append(now)

class AnomalySequenceDetector:
    """
    Detects predefined sequences of anomaly types using a fixed-size window.
    Sequences and responses are loaded from config ([AnomalySequences]).
    Uses tuple hashes for efficient matching.
    """
    def __init__(self, sequences_config, config):
        """
        Args:
            sequences_config: Config section or dict with 'sequences_of_interest' and 'suggested_responses'.
            config: Main config object.
        """
        self.config = config
        self.seq_window_size = config.getint('AnomalySequences', 'seq_window_size', fallback=3)
        self.seq_window = collections.deque(maxlen=self.seq_window_size)
        self.sequences = []
        self.seq_hashes = set()
        self.seq_hash_to_str = {}
        self.seq_hash_to_response = {}
        seqs = sequences_config.get('sequences_of_interest', '') if sequences_config else ''
        responses = sequences_config.get('suggested_responses', '') if sequences_config else ''
        # Parse sequences
        for seq in seqs.split(';'):
            parts = [s.strip() for s in seq.split('->') if s.strip()]
            if parts:
                self.sequences.append(parts)
                h = hash(tuple(parts))
                self.seq_hashes.add(h)
                self.seq_hash_to_str[h] = '->'.join(parts)
        # Parse suggested responses (format: X->Y->Z:response_dict;...)
        for resp in responses.split(';'):
            if ':' in resp:
                seq_str, resp_json = resp.split(':', 1)
                seq_parts = [s.strip() for s in seq_str.split('->') if s.strip()]
                h = hash(tuple(seq_parts))
                try:
                    import json
                    self.seq_hash_to_response[h] = json.loads(resp_json)
                except Exception:
                    self.seq_hash_to_response[h] = {'action': 'log_only'}
        self.last_match_hash = None
    def update(self, new_anomaly_type_str):
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
        current_hash = hash(tuple(self.seq_window))
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
    def __init__(self, config):
        """
        Args:
            config: Main config object.
        """
        self.config = config
        self.state = {'last_action_time': 0.0, 'actions_this_hour': 0, 'last_hour': 0}
    def execute_advanced_opportunity(self, opportunity_signal_dict, base_parameters_dict, current_persona_enum, config_dict):
        """
        Vectorized parameter modification for advanced opportunity actions.
        Applies safety gates and amplitude clamps from config ([OpportunitySafetyLimits]).
        Args:
            opportunity_signal_dict: Dict describing the opportunity signal.
            base_parameters_dict: Dict of base parameters (size, tp, sl, etc.).
            current_persona_enum: Current MarketPersona.
            config_dict: Full config object (should include 'OpportunitySafetyLimits').
        Returns:
            Dict describing the action taken or blocked.
        Side effects:
            Logs all actions and warnings for amplitude exceedance.
        """
        now = time.time()
        opp_type = opportunity_signal_dict.get('opportunity_type')
        persona = current_persona_enum.name if hasattr(current_persona_enum, 'name') else str(current_persona_enum)
        limits = config_dict['OpportunitySafetyLimits'] if 'OpportunitySafetyLimits' in config_dict else {}
        max_per_hour = int(limits.get('max_opportunity_actions_per_hour', 2))
        min_conf = float(limits.get('opportunity_min_base_confidence_floor', 0.3))
        max_env_score = float(limits.get('max_env_score_for_opp', 0.5)) if 'max_env_score_for_opp' in limits else 0.5
        cooldown = int(limits.get('opportunity_action_cooldown_seconds', 900))
        max_param_mod_amplitude = float(limits.get('max_param_mod_amplitude_factor', 2.0))
        # Hourly reset
        hour = int(now // 3600)
        if hour != self.state['last_hour']:
            self.state['actions_this_hour'] = 0
            self.state['last_hour'] = hour
        if self.state['actions_this_hour'] >= max_per_hour:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by hourly limit|type={opp_type}|persona={persona}")
            return {'action': 'blocked_hourly_limit'}
        if now - self.state['last_action_time'] < cooldown:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by cooldown|type={opp_type}|persona={persona}")
            return {'action': 'blocked_cooldown'}
        base_conf = base_parameters_dict.get('base_confidence', 0.0)
        env_score = base_parameters_dict.get('env_score', 0.0)
        if base_conf < min_conf:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by confidence floor|base_conf={base_conf:.3f}|floor={min_conf}")
            return {'action': 'blocked_confidence_floor'}
        if env_score > max_env_score:
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Blocked by env_score|env_score={env_score:.3f}|max={max_env_score}")
            return {'action': 'blocked_env_score'}
        # Vectorized parameter modification logic
        if opp_type == 'sharp_vwma_deviation' and persona == 'HUNTER':
            orig = np.array([
                base_parameters_dict.get('size', 1.0),
                base_parameters_dict.get('tp', 1.0),
                base_parameters_dict.get('sl', 1.0)
            ], dtype=np.float32)
            # Example: boost_factor based on VWMA deviation, persona aggression, and env stability
            vwma_dev = float(opportunity_signal_dict.get('vwma_deviation', 0.05))
            persona_agg = 1.1  # Could be persona-specific aggression factor
            env_stab = max(0.1, 1.0 - env_score)
            boost_factor = np.clip((vwma_dev * persona_agg) / env_stab, 1.0, 1.5)
            mods = orig * boost_factor
            # Clamp each mod to max_param_mod_amplitude_factor
            for i, name in enumerate(['size', 'tp', 'sl']):
                if mods[i] > orig[i] * max_param_mod_amplitude or mods[i] < orig[i] / max_param_mod_amplitude:
                    logger.warning(f"MAX_PARAM_MOD_AMPLITUDE_EXCEEDED|param={name}|original={orig[i]}|intended={mods[i]}")
            # Optionally, could clamp here, but for now just log
            logger.info(f"OPPORTUNITY_ADV_SHADOW: Would modify params (vectorized)|size={mods[0]:.2f}|tp={mods[1]:.2f}|sl={mods[2]:.2f}|type={opp_type}|persona={persona}|boost_factor={boost_factor:.3f}")
            self.state['last_action_time'] = now
            self.state['actions_this_hour'] += 1
            return {
                'action': 'modify_params',
                'original': {'size': float(orig[0]), 'tp': float(orig[1]), 'sl': float(orig[2])},
                'modified': {'size': float(mods[0]), 'tp': float(mods[1]), 'sl': float(mods[2])},
                'opportunity_type': opp_type,
                'persona': persona,
                'boost_factor': float(boost_factor)
            }
        return {'action': 'no_action'}

def detect_hunter_momentum_burst(price_series_np_array, config):
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
    threshold = config.getfloat('MarketPersona', 'hunter_momentum_burst_threshold_pct', fallback=0.03)
    windows = sliding_window_view(price_series_np_array, window_shape=5)
    # Calculate percent change for each window
    pct_changes = (windows[:, -1] - windows[:, 0]) / np.maximum(windows[:, 0], 1e-8)
    # Find 3 consecutive windows above threshold
    above = pct_changes > threshold
    # Use sliding window of 3 on above
    if len(above) < 3:
        return False
    burst_windows = sliding_window_view(above.astype(int), window_shape=3)
    for i, win in enumerate(burst_windows):
        if np.all(win):
            logger.info(f"HUNTER_MOMENTUM_BURST_DETECTED|start_idx={i}|threshold={threshold}")
            return True
    return False

def detect_hunter_vwap_momentum_pattern(current_price: float, vwap_or_vwma: float, price_velocity_1min: float, volume_ratio_1min_vs_5min: float, config: dict):
    """
    Detects the HUNTER persona's VWAP Momentum Pattern.
    Args:
        current_price: Latest price.
        vwap_or_vwma: Current VWAP or VWMA value.
        price_velocity_1min: 1-minute price change rate (e.g., (current_price - price_1_min_ago) / price_1_min_ago).
        volume_ratio_1min_vs_5min: Ratio of 1-min volume to 5-min average volume.
        config: Config object with [HunterPatterns] section.
    Returns:
        dict with pattern details if detected, else None.
    Side effects:
        Logs detection event.
    """
    vwap_distance_pct = (current_price - vwap_or_vwma) / vwap_or_vwma if vwap_or_vwma else 0.0
    min_dist = config.getfloat('HunterPatterns', 'vwap_momentum_min_distance_pct', fallback=0.002)
    min_vol_ratio = config.getfloat('HunterPatterns', 'vwap_momentum_min_volume_ratio', fallback=1.5)
    if abs(vwap_distance_pct) > min_dist:
        if np.sign(vwap_distance_pct) == np.sign(price_velocity_1min):
            if volume_ratio_1min_vs_5min > min_vol_ratio:
                direction = int(np.sign(vwap_distance_pct))
                max_conf = config.getfloat('HunterPatterns', 'vwap_max_pattern_confidence', fallback=0.95)
                dist_conf_mult = config.getfloat('HunterPatterns', 'vwap_dist_conf_multiplier', fallback=200)
                vol_conf_div = config.getfloat('HunterPatterns', 'vwap_vol_conf_divisor', fallback=3.0)
                calculated_confidence = min(max_conf, abs(vwap_distance_pct) * dist_conf_mult * volume_ratio_1min_vs_5min / vol_conf_div)
                logger.info(f"HUNTER_INSTINCT_PATTERN|type=vwap_momentum|direction={direction}|calculated_confidence={calculated_confidence:.3f}")
                return {'pattern_type': 'vwap_momentum', 'direction': direction, 'calculated_confidence': calculated_confidence}
    return None

def apply_guardian_tactical_validation(regime_signal_obj, current_env_score, recent_anomaly_reports, config):
    """
    MVP tactical validation for GUARDIAN persona. Returns a multiplier to apply to regime_signal_obj.confidence.
    Args:
        regime_signal_obj: RegimeSignal instance.
        current_env_score: float (smoothed env score).
        recent_anomaly_reports: list of AnomalyReport-like objects (should have .anomaly_strength and .timestamp attributes).
        config: config object.
    Returns:
        validation_multiplier (float, clamped).
    Side effects:
        Logs each check if triggered.
    """
    validation_multiplier = 1.0
    # Check 1: Regime-Environment Alignment
    high_vol_regime_name = config.get('GuardianTactics', 'high_vol_regime_name', fallback='high')
    poor_env_score_threshold = config.getfloat('GuardianTactics', 'poor_env_score_threshold', fallback=0.6)
    regime_env_mismatch_penalty_factor = config.getfloat('GuardianTactics', 'regime_env_mismatch_penalty_factor', fallback=0.8)
    if getattr(regime_signal_obj, 'volatility_regime', None) == high_vol_regime_name and current_env_score > poor_env_score_threshold:
        validation_multiplier *= regime_env_mismatch_penalty_factor
        logger.info(f"GUARDIAN_TACTIC|type=regime_env_mismatch|vol_regime={regime_signal_obj.volatility_regime}|env_score={current_env_score:.2f}|multiplier_applied={regime_env_mismatch_penalty_factor}")
    # Check 2: Recent Anomaly Interference
    anomaly_interference_lookback_seconds = config.getfloat('GuardianTactics', 'anomaly_interference_lookback_seconds', fallback=300)
    anomaly_interference_max_count = config.getint('GuardianTactics', 'anomaly_interference_max_count', fallback=2)
    anomaly_interference_penalty_factor = config.getfloat('GuardianTactics', 'anomaly_interference_penalty_factor', fallback=0.75)
    now = time.time()
    significant_anomalies = [a for a in recent_anomaly_reports if getattr(a, 'anomaly_strength', 0.0) > 0.5 and (now - getattr(a, 'timestamp', now)) < anomaly_interference_lookback_seconds]
    if len(significant_anomalies) > anomaly_interference_max_count:
        validation_multiplier *= anomaly_interference_penalty_factor
        logger.info(f"GUARDIAN_TACTIC|type=anomaly_interference|recent_anomaly_count={len(significant_anomalies)}|multiplier_applied={anomaly_interference_penalty_factor}")
    # Check 3: Basic Risk-Reward (Optional MVP)
    low_rr_min_trend = config.getfloat('GuardianTactics', 'low_rr_min_trend_for_good_rr', fallback=0.4)
    low_rr_confidence_threshold = config.getfloat('GuardianTactics', 'low_rr_confidence_threshold', fallback=0.3)
    low_rr_penalty_factor = config.getfloat('GuardianTactics', 'low_rr_penalty_factor', fallback=0.7)
    if abs(getattr(regime_signal_obj, 'trend_strength', 0.0)) < low_rr_min_trend and getattr(regime_signal_obj, 'confidence', 0.0) < low_rr_confidence_threshold:
        validation_multiplier *= low_rr_penalty_factor
        logger.info(f"GUARDIAN_TACTIC|type=low_risk_reward|trend={regime_signal_obj.trend_strength:.2f}|conf={regime_signal_obj.confidence:.2f}|multiplier_applied={low_rr_penalty_factor}")
    min_mult = config.getfloat('GuardianTactics', 'min_validation_multiplier', fallback=0.5)
    max_mult = config.getfloat('GuardianTactics', 'max_validation_multiplier', fallback=1.1)
    return float(np.clip(validation_multiplier, min_mult, max_mult))

class AdaptivePenaltyLearner:
    """
    Learns and adapts confidence penalties for each anomaly type based on historical L1 cycle PnL impact.
    Maintains a rolling history of PnL for each anomaly type and adjusts the base penalty accordingly.
    Configurable via [AdaptivePenaltyLearner] section in config.
    """
    def __init__(self, config):
        self.config = config
        # Structure: {anomaly_type_str: {'base_penalty': float, 'pnl_impact_history': deque}}
        self.penalty_profiles = defaultdict(self._default_penalty_profile)
    def _default_penalty_profile(self):
        initial_base_penalty = self.config.getfloat(
            'AdaptivePenaltyLearner',
            'initial_default_base_penalty',
            fallback=0.15
        )
        history_len = self.config.getint(
            'AdaptivePenaltyLearner',
            'pnl_impact_history_length',
            fallback=50
        )
        return {
            'base_penalty': initial_base_penalty,
            'pnl_impact_history': deque(maxlen=history_len)
        }
    def update_profile_with_l1_outcome(self, anomaly_type: str, simulated_pnl_l1_cycle: float):
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
        profile['pnl_impact_history'].append(simulated_pnl_l1_cycle)
        min_samples_for_adj = self.config.getint('AdaptivePenaltyLearner', 'min_pnl_samples_for_adjustment', fallback=10)
        if len(profile['pnl_impact_history']) < min_samples_for_adj:
            return
        recent_impact_lookback = self.config.getint('AdaptivePenaltyLearner', 'recent_pnl_impact_lookback', fallback=10)
        recent_pnl_list = list(profile['pnl_impact_history'])[-recent_impact_lookback:]
        if not recent_pnl_list:
            return
        avg_recent_impact = np.mean(recent_pnl_list)
        old_penalty = profile['base_penalty']
        benign_impact_threshold = self.config.getfloat('AdaptivePenaltyLearner', 'benign_pnl_impact_threshold', fallback=-0.01)
        dangerous_impact_threshold = self.config.getfloat('AdaptivePenaltyLearner', 'dangerous_pnl_impact_threshold', fallback=-0.05)
        penalty_reduction_factor = self.config.getfloat('AdaptivePenaltyLearner', 'penalty_reduction_factor', fallback=0.95)
        penalty_increase_factor = self.config.getfloat('AdaptivePenaltyLearner', 'penalty_increase_factor', fallback=1.10)
        min_penalty_cap = self.config.getfloat('AdaptivePenaltyLearner', 'min_penalty_cap', fallback=0.01)
        max_penalty_cap = self.config.getfloat('AdaptivePenaltyLearner', 'max_penalty_cap', fallback=0.5)
        if avg_recent_impact > benign_impact_threshold:
            profile['base_penalty'] *= penalty_reduction_factor
        elif avg_recent_impact < dangerous_impact_threshold:
            profile['base_penalty'] *= penalty_increase_factor
        profile['base_penalty'] = np.clip(profile['base_penalty'], min_penalty_cap, max_penalty_cap)
        if abs(old_penalty - profile['base_penalty']) > 1e-4:
            logger.info(f"ADAPTIVE_PENALTY_UPDATE|anomaly_type={anomaly_type}|old_penalty={old_penalty:.3f}|new_penalty={profile['base_penalty']:.3f}|avg_recent_pnl={avg_recent_impact:.3f}|pnl_history_len={len(profile['pnl_impact_history'])}")
    def get_learned_penalty(self, anomaly_type: str) -> float:
        """
        Returns the learned base_penalty for the anomaly_type.
        If never seen, returns the initial default penalty.
        """
        return self.penalty_profiles[anomaly_type]['base_penalty']

class PersonaEffectivenessTracker:
    """
    Tracks persona effectiveness (simulated win rate) by environmental context and provides a bias factor for persona selection.
    Uses a Bayesian-inspired update for success rate and calculates a bias factor for each persona/env context.
    Configurable via [PersonaEffectiveness] section in config.
    """
    def __init__(self, config):
        self.config = config
        # Structure: {persona_name_str: {env_context_str: {'outcomes': deque, 'success_rate_estimate': float, 'total_observations': int}}}
        self.persona_outcomes_by_context = defaultdict(lambda: defaultdict(self._default_context_profile))
        self.env_score_bins = [
            config.getfloat('PersonaEffectiveness', 'env_score_bin_low_medium_threshold', fallback=0.3),
            config.getfloat('PersonaEffectiveness', 'env_score_bin_medium_high_threshold', fallback=0.6)
        ]
        self.min_obs_for_bias = config.getint('PersonaEffectiveness', 'min_observations_for_bias', fallback=10)
        self.bias_strength_factor = config.getfloat('PersonaEffectiveness', 'persona_bias_strength_factor', fallback=0.05)
    def _default_context_profile(self):
        history_len = self.config.getint('PersonaEffectiveness', 'outcome_history_length_per_context', fallback=30)
        return {'outcomes': deque(maxlen=history_len), 'success_rate_estimate': 0.5, 'total_observations': 0}
    def _get_env_context_str(self, env_score: float) -> str:
        if env_score < self.env_score_bins[0]: return "low_env_score"
        if env_score < self.env_score_bins[1]: return "medium_env_score"
        return "high_env_score"
    def record_l1_outcome(self, persona, env_score: float, simulated_pnl: float):
        """
        Record the outcome of an L1 cycle for a persona in a given env context.
        Args:
            persona: MarketPersona enum.
            env_score: Environmental score at the time.
            simulated_pnl: Simulated PnL (+1/-1/0).
        Side effects:
            Updates internal stats and logs update.
        """
        persona_name = persona.name
        env_context_str = self._get_env_context_str(env_score)
        profile = self.persona_outcomes_by_context[persona_name][env_context_str]
        profile['outcomes'].append(1 if simulated_pnl > 0 else 0)
        profile['total_observations'] += 1
        # Bayesian update with uniform prior Beta(1,1)
        successes = sum(profile['outcomes'])
        total_samples = len(profile['outcomes'])
        profile['success_rate_estimate'] = (1 + successes) / (2 + total_samples)
        logger.debug(f"PERSONA_EFFECTIVENESS_UPDATE|persona={persona_name}|env_context={env_context_str}|pnl={simulated_pnl:.2f}|new_success_rate_est={profile['success_rate_estimate']:.3f}|obs={profile['total_observations']}")
    def get_persona_bias_factor(self, persona_to_evaluate, current_env_score: float) -> float:
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
        persona_name = persona_to_evaluate.name
        env_context_str = self._get_env_context_str(current_env_score)
        profile = self.persona_outcomes_by_context[persona_name][env_context_str]
        if profile['total_observations'] < self.min_obs_for_bias:
            return 1.0
        effectiveness_delta = profile['success_rate_estimate'] - 0.5
        bias = 1.0 + (effectiveness_delta * 2 * self.bias_strength_factor)
        min_bias = self.config.getfloat('PersonaEffectiveness', 'min_persona_bias_factor', fallback=0.9)
        max_bias = self.config.getfloat('PersonaEffectiveness', 'max_persona_bias_factor', fallback=1.1)
        clamped_bias = np.clip(bias, min_bias, max_bias)
        if abs(clamped_bias - 1.0) > 1e-4:
            logger.info(f"PERSONA_BIAS_CALCULATED|persona={persona_name}|env_context={env_context_str}|success_rate_est={profile['success_rate_estimate']:.3f}|bias_factor={clamped_bias:.3f}")
        return clamped_bias

# === MVP Layered Decision Architecture & Ensemble Confidence Detector ===
class InstinctiveLayer:
    def __init__(self, config):
        self.config = config
    def evaluate(self, market_state_snapshot, persona_context_dict):
        logger.info("InstinctiveLayer.evaluate called | persona=%s", persona_context_dict.get('persona'))
        # MVP: return neutral
        return {'action_bias': None, 'confidence_mod': 0.0}

class TacticalLayer:
    def __init__(self, config):
        self.config = config
    def evaluate(self, market_state_snapshot, persona_context_dict):
        logger.info("TacticalLayer.evaluate called | persona=%s", persona_context_dict.get('persona'))
        return {'action_bias': None, 'confidence_mod': 0.0}

class StrategicLayer:
    def __init__(self, config):
        self.config = config
    def evaluate(self, market_state_snapshot, persona_context_dict):
        logger.info("StrategicLayer.evaluate called | persona=%s", persona_context_dict.get('persona'))
        return {'action_bias': None, 'confidence_mod': 0.0}

class LayeredDecisionArchitecture:
    def __init__(self, config):
        self.config = config
        self.instinctive = InstinctiveLayer(config)
        self.tactical = TacticalLayer(config)
        self.strategic = StrategicLayer(config)
    def get_layer_weights(self, persona):
        persona_str = persona.name.lower()
        section = 'DecisionLayers'
        return {
            'instinctive': self.config.getfloat(section, f'{persona_str}_weight_instinctive', fallback=0.33),
            'tactical': self.config.getfloat(section, f'{persona_str}_weight_tactical', fallback=0.33),
            'strategic': self.config.getfloat(section, f'{persona_str}_weight_strategic', fallback=0.34),
        }
    def process_l1_decision(self, market_state_snapshot, active_persona, base_regime_signal, shadow_mode=True):
        weights = self.get_layer_weights(active_persona)
        persona_ctx = {'persona': active_persona.name}
        decisions = {
            'instinctive': self.instinctive.evaluate(market_state_snapshot, persona_ctx),
            'tactical': self.tactical.evaluate(market_state_snapshot, persona_ctx),
            'strategic': self.strategic.evaluate(market_state_snapshot, persona_ctx),
        }
        logger.info(f"LAYERED_DECISION_ARCH|persona={active_persona.name}|weights={weights}|decisions={decisions}")
        # MVP: average confidence_mods, log action_biases
        avg_conf_mod = np.mean([d.get('confidence_mod', 0.0) for d in decisions.values()])
        if not shadow_mode:
            old_conf = base_regime_signal.confidence
            base_regime_signal.confidence = float(np.clip(base_regime_signal.confidence + avg_conf_mod, 0.0, 1.0))
            logger.info(f"LAYERED_DECISION_APPLIED|old_conf={old_conf:.3f}|avg_conf_mod={avg_conf_mod:.3f}|new_conf={base_regime_signal.confidence:.3f}")
        else:
            would_be_conf = float(np.clip(base_regime_signal.confidence + avg_conf_mod, 0.0, 1.0))
            logger.info(f"LAYERED_DECISION_SHADOW|avg_conf_mod={avg_conf_mod:.3f}|current_conf={base_regime_signal.confidence:.3f}|would_be_conf={would_be_conf:.3f}")
        return base_regime_signal

class EnsembleConfidenceDetector:
    def __init__(self, config):
        self.config = config
    def detect_super_confidence_conditions(self, system_state_snapshot):
        # MVP: check for hunter_confluence pattern
        persona = system_state_snapshot.get('current_persona')
        regime_signal = system_state_snapshot.get('regime_signal')
        env_score = system_state_snapshot.get('env_score')
        # For MVP, just check if persona is HUNTER and regime_signal.confidence > threshold and env_score < max
        if persona and persona.name == 'HUNTER':
            min_conv = self.config.getfloat('EnsembleConfidence', 'hunter_confluence_regime_conviction_min', fallback=0.7)
            max_env = self.config.getfloat('EnsembleConfidence', 'hunter_confluence_max_env_stress', fallback=0.4)
            boost_mult = self.config.getfloat('EnsembleConfidence', 'hunter_confluence_boost_multiplier', fallback=1.2)
            boost_dur = self.config.getint('EnsembleConfidence', 'hunter_confluence_boost_duration_l1_cycles', fallback=3)
            if regime_signal and getattr(regime_signal, 'confidence', 0.0) > min_conv and env_score is not None and env_score < max_env:
                logger.info(f"ENSEMBLE_CONFIDENCE_DETECTED|type=hunter_confluence|multiplier={boost_mult}|duration={boost_dur}")
                return {'boost_type': 'hunter_confluence', 'confidence_multiplier': boost_mult, 'duration_cycles': boost_dur}
        return None 