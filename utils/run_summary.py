import time
import json
import os
from collections import defaultdict, deque
from datetime import datetime
import numpy as np
from enum import Enum
import logging
import array

logger = logging.getLogger(__name__)

class RunSummaryGenerator:
    """
    Tracks and aggregates all key metrics, events, and outcomes for a bot run.
    Generates a detailed run_summary.json for post-run analysis, including persona effectiveness, hourly snapshots, anomaly distribution by env context, shadowed opportunity analysis, and key observations.
    """
    def __init__(self, config):
        """
        Args:
            config: Main config object (should include [RunSummary] for maxlen, etc.)
        """
        self.config = config
        self.start_time = time.time()
        self.l0_polls = 0
        self.l1_cycles_triggered = 0
        # Persona Dynamics
        self.persona_transitions_log = [] # list of dicts: {'timestamp', 'from', 'to', 'reason'}
        self.current_persona_for_summary = None
        self.last_persona_transition_time = self.start_time
        self.persona_time_spent = defaultdict(float)
        # Env Health
        self.env_score_history = array.array('f')  # stores only scores, not timestamps, for memory efficiency
        self.env_score_time_history = array.array('d')  # store timestamps separately if needed
        self.env_score_history_maxlen = config.getint('RunSummary', 'env_score_history_maxlen', fallback=1000)
        # Defense Activations
        self.circuit_breaker_activations = 0
        self.persona_transition_blocks = 0
        # Anomaly Intel
        self.anomalies_detected_by_type = defaultdict(int)
        # New: anomaly distribution by env context
        self.anomaly_distribution_by_env_context = defaultdict(lambda: defaultdict(int))  # env_context -> anomaly_type -> count
        self.opportunities_identified = 0
        self.opportunities_acted_upon_shadowed = 0
        # New: shadowed opportunity outcomes
        self.shadowed_opportunity_outcomes = []  # list of dicts: {'opportunity_type', 'suggested_action_details', 'subsequent_simulated_pnl'}
        # System Stability
        self.error_count = 0
        # Persona Effectiveness
        self.l1_outcomes = defaultdict(list) # persona_name -> list of (sim_pnl, conf, env_score, action_str)
        # For env context binning
        self.env_score_bins = [
            config.getfloat('PersonaEffectiveness', 'env_score_bin_low_medium_threshold', fallback=0.3),
            config.getfloat('PersonaEffectiveness', 'env_score_bin_medium_high_threshold', fallback=0.6)
        ]
    def _get_env_context_str(self, env_score: float) -> str:
        if env_score < self.env_score_bins[0]: return "low_env_score"
        if env_score < self.env_score_bins[1]: return "medium_env_score"
        return "high_env_score"
    def track_l1_outcome(self, persona_name_str, simulated_pnl_float, regime_confidence_float, env_score_float, action_str=None):
        """
        Record the outcome of an L1 cycle for persona effectiveness analysis.
        Args:
            persona_name_str: Name of the persona.
            simulated_pnl_float: Simulated PnL (+1/-1/0 for MVP logic).
            regime_confidence_float: Confidence at entry.
            env_score_float: Environmental score at entry.
            action_str: Action taken (BUY/SELL/HOLD).
        """
        self.l1_outcomes[persona_name_str].append((simulated_pnl_float, regime_confidence_float, env_score_float, action_str))
    def log_l0_poll(self): self.l0_polls += 1
    def log_l1_cycle(self): self.l1_cycles_triggered += 1
    def log_persona_transition(self, old_p_name: str, new_p_name: str, reason: str, env_score: float):
        now = time.time()
        if self.current_persona_for_summary is not None:
            self.persona_time_spent[self.current_persona_for_summary] += (now - self.last_persona_transition_time)
        self.persona_transitions_log.append({
            'timestamp': now, 'from': old_p_name, 'to': new_p_name, 
            'reason': reason, 'env_score': env_score
        })
        self.current_persona_for_summary = new_p_name
        self.last_persona_transition_time = now
    def log_env_score(self, score_and_time):
        score, t = score_and_time
        if len(self.env_score_history) >= self.env_score_history_maxlen:
            self.env_score_history.pop(0)
            self.env_score_time_history.pop(0)
        self.env_score_history.append(score)
        self.env_score_time_history.append(t)
    def log_circuit_breaker_trigger(self): self.circuit_breaker_activations += 1
    def log_persona_transition_blocked(self): self.persona_transition_blocks += 1
    def log_anomaly_detected(self, anomaly_type: str, current_smoothed_env_score: float = None):
        if anomaly_type: self.anomalies_detected_by_type[anomaly_type] += 1
        # New: also track by env context if env_score provided
        if anomaly_type and current_smoothed_env_score is not None:
            env_context = self._get_env_context_str(current_smoothed_env_score)
            self.anomaly_distribution_by_env_context[env_context][anomaly_type] += 1
    def log_opportunity_identified(self, opportunity_result: dict):
        if opportunity_result and opportunity_result.get("opportunity_type") != "none":
            self.opportunities_identified += 1
            if opportunity_result.get("action_taken_shadowed"):
                self.opportunities_acted_upon_shadowed +=1
    def log_shadowed_opportunity_outcome(self, opportunity_type: str, suggested_action_details: dict, subsequent_simulated_pnl: float):
        """
        Log the outcome of a shadowed opportunity signal for post-hoc analysis.
        Args:
            opportunity_type: str
            suggested_action_details: dict
            subsequent_simulated_pnl: float (PnL of the L1 cycle that would have been affected)
        """
        self.shadowed_opportunity_outcomes.append({
            'opportunity_type': opportunity_type,
            'suggested_action_details': suggested_action_details,
            'subsequent_simulated_pnl': subsequent_simulated_pnl
        })
    def log_error(self):
        """
        Increment the error count (for system_stability tracking).
        Call this from global exception handler or critical error points.
        """
        self.error_count += 1
    def _build_summary_dict(self):
        """
        Aggregate all tracked metrics and return a detailed summary dictionary.
        Includes persona effectiveness, hourly micro-climate, and key observations.
        """
        # Finalize time spent in the last persona
        if self.current_persona_for_summary is not None:
            self.persona_time_spent[self.current_persona_for_summary] += (time.time() - self.last_persona_transition_time)
        runtime_seconds = time.time() - self.start_time
        runtime_hours = runtime_seconds / 3600.0
        # Persona effectiveness
        persona_effectiveness = {}
        for persona, outcomes in self.l1_outcomes.items():
            if outcomes:
                pnls = [o[0] for o in outcomes]
                confs = [o[1] for o in outcomes]
                actions = [o[3] for o in outcomes if len(o) > 3]
                wins = [1 for o in outcomes if o[0] > 0]
                total = len(outcomes)
                win_rate = float(sum(wins)) / total if total > 0 else 0.0
                persona_effectiveness[persona] = {
                    'avg_simulated_pnl': float(np.mean(pnls)),
                    'simulated_win_rate': win_rate,
                    'total_l1_cycles_for_persona': total,
                    'avg_entry_confidence': float(np.mean(confs)),
                }
        # Micro-climate hourly aggregation
        hourly_snapshots = []
        if self.env_score_history:
            start_ts = self.start_time
            end_ts = time.time()
            hour = 0
            while True:
                hour_start = start_ts + hour * 3600
                hour_end = hour_start + 3600
                if hour_start >= end_ts:
                    break
                # Use env_score_time_history to filter scores in hour
                scores_in_hour = [self.env_score_history[i] for i in range(len(self.env_score_history)) if hour_start <= self.env_score_time_history[i] < hour_end]
                avg_env = float(np.mean(scores_in_hour)) if scores_in_hour else None
                # Dominant persona for hour
                persona_times = {}
                for t in self.persona_transitions_log:
                    if hour_start <= t['timestamp'] < hour_end:
                        persona = t['to']
                        persona_times[persona] = persona_times.get(persona, 0) + 1
                dominant_persona = max(persona_times, key=persona_times.get) if persona_times else None
                # Anomalies in hour (approximate by count in persona_transitions_log for now)
                total_anomalies = 0 # Could be improved if anomaly timestamps are tracked
                hourly_snapshots.append({
                    'hour_start_iso': datetime.fromtimestamp(hour_start).isoformat(),
                    'avg_env_score_this_hour': avg_env,
                    'dominant_persona_this_hour': dominant_persona,
                    'total_anomalies_this_hour': total_anomalies
                })
                hour += 1
        # Key observations (now enhanced)
        key_observations = self._generate_key_insights(runtime_hours)
        # New: anomaly distribution by env context
        anomaly_dist_by_env = {ctx: dict(types) for ctx, types in self.anomaly_distribution_by_env_context.items()}
        # New: shadowed opportunity analysis
        shadowed_opp_analysis = {}
        if self.shadowed_opportunity_outcomes:
            from collections import defaultdict
            opp_stats = defaultdict(lambda: {'count_identified': 0, 'count_would_have_been_profitable': 0, 'simulated_win_rate_if_acted': 0.0})
            for entry in self.shadowed_opportunity_outcomes:
                opp_type = entry['opportunity_type']
                pnl = entry['subsequent_simulated_pnl']
                opp_stats[opp_type]['count_identified'] += 1
                if pnl > 0:
                    opp_stats[opp_type]['count_would_have_been_profitable'] += 1
            for opp_type, stats in opp_stats.items():
                total = stats['count_identified']
                wins = stats['count_would_have_been_profitable']
                stats['simulated_win_rate_if_acted'] = float(wins) / total if total > 0 else 0.0
            shadowed_opp_analysis = dict(opp_stats)
        summary = {
            'run_metadata': {'start_time_iso': datetime.fromtimestamp(self.start_time).isoformat(), 
                             'end_time_iso': datetime.now().isoformat(), 
                             'runtime_hours': round(runtime_hours, 2)},
            'cycle_counts': {'L0_polls': self.l0_polls, 'L1_cycles': self.l1_cycles_triggered,
                             'L1_per_1000_L0': round((self.l1_cycles_triggered / (self.l0_polls or 1)) * 1000, 2)},
            'persona_dynamics': {
                'transitions_count': len(self.persona_transitions_log),
                'time_spent_seconds_per_persona': dict(self.persona_time_spent),
                'transitions_detail': self.persona_transitions_log[-self.config.getint('RunSummary', 'max_transitions_in_summary', fallback=20):]
            },
            'environmental_health': {
                'env_score_min': round(min(self.env_score_history), 3) if self.env_score_history else None,
                'env_score_max': round(max(self.env_score_history), 3) if self.env_score_history else None,
                'env_score_avg': round(np.mean(self.env_score_history), 3) if self.env_score_history else None,
                'env_score_std': round(np.std(self.env_score_history), 3) if self.env_score_history else None,
            },
            'persona_effectiveness': persona_effectiveness,
            'hourly_snapshots': hourly_snapshots,
            'defense_activations': {
                'anomaly_circuit_breaker_triggers': self.circuit_breaker_activations,
                'persona_transition_blocks': self.persona_transition_blocks
            },
            'anomaly_intelligence': {
                'anomalies_detected_by_type': dict(self.anomalies_detected_by_type),
                'anomaly_distribution_by_env_context': anomaly_dist_by_env,
                'opportunities_identified_shadowed': self.opportunities_identified,
                'shadowed_opportunity_analysis': shadowed_opp_analysis
            },
            'system_stability': {'error_count': self.error_count},
            'key_observations_generated': key_observations
        }
        return summary
    def generate_and_save_summary(self, output_path="run_summary.json"):
        """
        Generates and saves the run summary as a JSON file, ensuring all values are JSON serializable.
        """
        summary = self._build_summary_dict()
        # Sanitize all NumPy types to standard Python types for JSON serialization
        summary = _sanitize_for_json(summary)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
    def _generate_key_insights(self, runtime_hours):
        """
        Generate key insights/observations based on tracked data, including new analytics.
        """
        insights = []
        # Existing logic
        if self.circuit_breaker_activations > (runtime_hours * 5):
            insights.append("High rate of Anomaly Circuit Breaker activations detected.")
        if len(self.persona_transitions_log) > (runtime_hours * 20):
            insights.append("High rate of Persona transitions observed.")
        # New: anomaly distribution by env context
        for env_ctx, type_counts in self.anomaly_distribution_by_env_context.items():
            for anomaly_type, count in type_counts.items():
                if count > 5:  # Simple threshold for "frequent"
                    insights.append(f"Insight: {anomaly_type} frequently occurred during {env_ctx.replace('_', ' ')}.")
        # New: shadowed opportunity win rates
        if self.shadowed_opportunity_outcomes:
            from collections import Counter
            opp_types = [entry['opportunity_type'] for entry in self.shadowed_opportunity_outcomes]
            opp_counts = Counter(opp_types)
            for opp_type, count in opp_counts.items():
                wins = sum(1 for entry in self.shadowed_opportunity_outcomes if entry['opportunity_type'] == opp_type and entry['subsequent_simulated_pnl'] > 0)
                win_rate = float(wins) / count if count > 0 else 0.0
                if count >= 5 and win_rate > 0.6:
                    insights.append(f"Insight: Shadowed opportunity {opp_type} showed a promising simulated win rate of {int(win_rate*100)}%.")
        return insights 

def _sanitize_for_json(obj):
    """
    Recursively convert all NumPy float/int types in a dict/list to standard Python float/int for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, float) or isinstance(obj, int) or obj is None:
        return obj
    else:
        return obj 