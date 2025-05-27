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
from src.meta_parameter_monitor import MetaParameterMonitor
import settings  # <-- Add this import
import json
from src.lee import LEE
from src.market_persona import MarketPersona
from src.logic_dna import LogicDNA_v1
from src.mle_engine import MLE_v0_1
from src.ces_module import CES_v1_0

logger = logging.getLogger(__name__)

MAX_CANDIDATE_SLOTS = 2
CANDIDATE_VIRTUAL_CAPITAL = 10.0  # Notional, larger than pool
CANDIDATE_MIN_SHARPE = 0.3
CANDIDATE_MIN_PNL = 5.0
CANDIDATE_MIN_TICKS = 100
CANDIDATE_MIN_ACTIVATIONS = 3
CANDIDATE_EPSILON = 1e-6
CANDIDATE_MAX_LIFE = 2000  # ticks
CANDIDATE_SHARPE_DROP_TRIGGER = 0.3  # 30% drop triggers self-calibration
AB_TEST_TICKS = 200  # Configurable A/B test period
FITNESS_WEIGHTS = {'sharpe': 0.4, 'regime_consistency': 0.3, 'diversification': 0.3}
AB_SHARPE_THRESHOLD = 0.10  # 10% Sharpe improvement
AB_PNL_THRESHOLD = 0.05     # 5% PnL improvement
INFLUENCE_MULTIPLIER_MAX = 1.5
INFLUENCE_MULTIPLIER_MIN = 0.5
INFLUENCE_MULTIPLIER_STEP_UP = 0.1
INFLUENCE_MULTIPLIER_STEP_DOWN = 0.1
INFLUENCE_SCORE_HIGH_PCT = 0.75  # Top 25% get boost
INFLUENCE_SCORE_LOW_PCT = 0.25   # Bottom 25% get reduced
COOLDOWN_TICKS_AFTER_TWEAK = 750

def check_settings_dict(settings_dict, required_keys, dict_name):
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        raise RuntimeError(f"CRITICAL: Missing keys in {dict_name}: {missing}")

class ConflictResolver:
    """
    MVP stub for conflict detection and resolution between system influences.
    For now, always returns False for conflicts and passes through parameters.
    """
    def __init__(self, config):
        self.config = config

    def detect_conflicts(self, current_params: dict, influences: dict) -> bool:
        """
        Detects if there are conflicting influences on key parameters.
        For MVP, always returns False.
        """
        logger.debug("ConflictResolver.detect_conflicts called (MVP: always returns False)")
        return False

    def resolve(self, current_params: dict, influences: dict) -> dict:
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
    def __init__(self, window=50):
        self.window = window
        self.price_buffer = deque(maxlen=window)
        self.last_regime = None

    def update(self, price):
        self.price_buffer.append(price)
        return self.classify()

    def classify(self):
        if len(self.price_buffer) < 10:
            return None
        prices = np.array(self.price_buffer)
        returns = np.diff(prices)
        vol = np.std(returns)
        sma = np.convolve(prices, np.ones(20)/20, mode='valid')
        trend = (sma[-1] - sma[0]) / 20 if len(sma) >= 2 else 0.0
        # Regime logic
        if vol > 1.0:
            vol_regime = 'HIGH_VOL'
        elif vol < 0.3:
            vol_regime = 'LOW_VOL'
        else:
            vol_regime = 'MEDIUM_VOL'
        if trend > 0.05:
            trend_regime = 'UPTREND_STRONG'
        elif trend < -0.05:
            trend_regime = 'DOWNTREND_STRONG'
        else:
            trend_regime = 'SIDEWAYS_CHOPPY'
        regime = {'VOLATILITY': vol_regime, 'TREND': trend_regime}
        if regime != self.last_regime:
            log_event(
                'REGIME_CHANGE_DETECTED',
                {
                    'old_regime': self.last_regime,
                    'new_regime': regime,
                    'timestamp': datetime.now().isoformat(),
                }
            )
            logger.info(f"SO RegimeClassifier: Regime changed to {regime}")
            self.last_regime = regime
        return regime

class CandidateStrategySlot:
    """
    Represents a candidate strategy slot in the SO, tracking a DNA's performance and lifecycle.
    Args:
        pool_entry (PoolDNAEntry): The graduated pool entry containing the DNA.
    Attributes:
        dna (LogicDNA): The DNA instance (with unique id).
        ... (other attributes as before)
    """
    def __init__(self, pool_entry):
        self.dna = pool_entry.dna
        self.dedicated_virtual_capital_candidate = SO_SETTINGS['CANDIDATE_VIRTUAL_CAPITAL']
        self.pnl_candidate = 0.0
        self.sharpe_candidate = 0.0
        self.ticks_in_candidate = 0
        self.activation_count_candidate = 0
        self.pnl_history = []
        self.logger = logging.getLogger(__name__)
        self.ready_for_demotion = False
        self.recent_high_sharpe = 0.0
        self.self_calibration_needed = False
        self.influence_multiplier = 1.0
        self.last_tweak_implemented_at = -SO_SETTINGS['COOLDOWN_TICKS_AFTER_TWEAK']  # So it's not in cooldown at start

    def evaluate_tick(self, market_tick, global_tick=None):
        """
        Evaluate this candidate's DNA for the current market tick, updating PnL, Sharpe, and calibration triggers.
        """
        self.ticks_in_candidate += 1
        indicator_value = market_tick.get(self.dna.trigger_indicator, None)
        if indicator_value is None:
            return
        op = self.dna.trigger_operator
        threshold = self.dna.trigger_threshold
        triggered = (op == '<' and indicator_value < threshold) or (op == '>' and indicator_value > threshold)
        if triggered:
            self.activation_count_candidate += 1
            price_now = market_tick.get('price', 0.0)
            price_prev = market_tick.get('price_prev', price_now)
            if self.dna.action_target.startswith('buy'):
                pnl = self.dedicated_virtual_capital_candidate * (1.0 if price_now > price_prev else -1.0)
            elif self.dna.action_target.startswith('sell'):
                pnl = self.dedicated_virtual_capital_candidate * (1.0 if price_now < price_prev else -1.0)
            else:
                pnl = 0.0
            self.pnl_candidate += pnl
            self.pnl_history.append(pnl)
            self.logger.info(f"CandidateSlot: DNA {self.dna.id} triggered | PnL: {pnl} | TotalPnL: {self.pnl_candidate}")
        # Update Sharpe
        if self.pnl_history:
            mean_pnl = sum(self.pnl_history) / len(self.pnl_history)
            variance = sum((x - mean_pnl) ** 2 for x in self.pnl_history) / len(self.pnl_history)
            stddev = math.sqrt(variance) + SO_SETTINGS['CANDIDATE_EPSILON']
            self.sharpe_candidate = mean_pnl / stddev
            # Track recent high Sharpe
            if self.sharpe_candidate > self.recent_high_sharpe:
                self.recent_high_sharpe = self.sharpe_candidate
            # Self-calibration trigger
            if (self.recent_high_sharpe > 0 and self.sharpe_candidate < (1 - SO_SETTINGS['CANDIDATE_SHARPE_DROP_TRIGGER']) * self.recent_high_sharpe and
                (global_tick is None or global_tick - self.last_tweak_implemented_at > SO_SETTINGS['COOLDOWN_TICKS_AFTER_TWEAK'])):
                self.self_calibration_needed = True
                self.logger.info(f"CandidateSlot: DNA {self.dna.id} triggered self-calibration | Sharpe dropped from {self.recent_high_sharpe:.2f} to {self.sharpe_candidate:.2f}")
        # Demotion check
        if self.ticks_in_candidate > SO_SETTINGS['CANDIDATE_MAX_LIFE'] and (self.sharpe_candidate < SO_SETTINGS['CANDIDATE_MIN_SHARPE'] or self.pnl_candidate < SO_SETTINGS['CANDIDATE_MIN_PNL']):
            self.ready_for_demotion = True
            self.logger.info(f"CandidateSlot: DNA {self.dna.id} flagged for demotion | Sharpe={self.sharpe_candidate:.2f} | PnL={self.pnl_candidate:.2f}")

    def is_ready_for_graduation(self):
        """
        Check if this candidate is ready for graduation based on performance and activity.
        Returns:
            bool: True if ready for graduation.
        """
        return (self.ticks_in_candidate >= SO_SETTINGS['CANDIDATE_MIN_TICKS'] and
                self.activation_count_candidate >= SO_SETTINGS['CANDIDATE_MIN_ACTIVATIONS'] and
                self.pnl_candidate >= SO_SETTINGS['CANDIDATE_MIN_PNL'] and
                self.sharpe_candidate >= SO_SETTINGS['CANDIDATE_MIN_SHARPE'])

    def run_self_calibration(self, recent_market_data, so, regime=None):
        """
        Parameter Explorer & Tweak Assessor: Generate 2-3 tweaks, micro-sim each, compare to original, propose best to SO.
        Args:
            recent_market_data: List of dicts (market ticks) for micro-sim.
            so: SystemOrchestrator instance for logging proposal.
            regime: Regime information for micro-sim.
        """
        self.logger.info(f"CandidateSlot: Running self-calibration for DNA {self.dna.id}")
        tweaks = [mutate_dna(self.dna, mutation_strength=0.1) for _ in range(3)]
        results = []
        sensitivity_profile = []
        # Sim original
        orig_result = run_nanostrat_test(self.dna, recent_market_data)
        results.append({'dna': self.dna, 'result': orig_result, 'is_original': True})
        # Sim tweaks (record param, value, result)
        for t in tweaks:
            t_result = run_nanostrat_test(t, recent_market_data)
            results.append({'dna': t, 'result': t_result, 'is_original': False})
            # Sensitivity: which param changed?
            for param in ['trigger_threshold', 'action_value']:
                if getattr(t, param) != getattr(self.dna, param):
                    sensitivity_profile.append({'param': param, 'value': getattr(t, param), 'pnl': t_result['virtual_pnl'], 'sharpe': t_result['virtual_pnl'] / (np.std([t_result['virtual_pnl']]) + 1e-6)})
        # Find best
        best = max(results, key=lambda x: x['result']['virtual_pnl'])
        if not best['is_original'] and best['result']['virtual_pnl'] > orig_result['virtual_pnl']:
            self.logger.info(f"CandidateSlot: Best tweak found for DNA {self.dna.id}: {best['dna']} | PnL uplift: {best['result']['virtual_pnl'] - orig_result['virtual_pnl']:.2f}")
            micro_sim_regimes = regime if regime else 'unknown'
            self.propose_tweak_to_so(best['dna'], self.dna, best['result'], orig_result, so, micro_sim_regimes, sensitivity_profile)
        else:
            self.logger.info(f"CandidateSlot: No tweak outperformed original for DNA {self.dna.id}")
        self.self_calibration_needed = False

    def propose_tweak_to_so(self, tweak_dna, orig_dna, tweak_result, orig_result, so, micro_sim_regimes=None, sensitivity_profile=None):
        so.log_tweak_proposal(
            candidate_id=orig_dna.id,
            orig_params={
                'trigger_indicator': orig_dna.trigger_indicator,
                'trigger_operator': orig_dna.trigger_operator,
                'trigger_threshold': orig_dna.trigger_threshold,
                'context_regime_id': orig_dna.context_regime_id,
                'action_target': orig_dna.action_target,
                'action_type': orig_dna.action_type,
                'action_value': orig_dna.action_value,
                'resource_cost': getattr(orig_dna, 'resource_cost', 1)
            },
            tweak_params={
                'trigger_indicator': tweak_dna.trigger_indicator,
                'trigger_operator': tweak_dna.trigger_operator,
                'trigger_threshold': tweak_dna.trigger_threshold,
                'context_regime_id': tweak_dna.context_regime_id,
                'action_target': tweak_dna.action_target,
                'action_type': tweak_dna.action_type,
                'action_value': tweak_dna.action_value,
                'resource_cost': getattr(tweak_dna, 'resource_cost', 1)
            },
            orig_result=orig_result,
            tweak_result=tweak_result,
            micro_sim_regimes=micro_sim_regimes,
            sensitivity_profile=sensitivity_profile
        )

class ABTestManager:
    def __init__(self):
        self.active_tests = {}  # key: candidate_id, value: ABTest

    class ABTest:
        def __init__(self, candidate_id, control_dna, variant_dna, start_tick, regime):
            self.candidate_id = candidate_id
            self.control_dna = control_dna
            self.variant_dna = variant_dna
            self.start_tick = start_tick
            self.regime = regime
            self.ticks = 0
            self.control_metrics = {'pnl': 0.0, 'activations': 0, 'pnl_history': []}
            self.variant_metrics = {'pnl': 0.0, 'activations': 0, 'pnl_history': []}
            self.finished = False

        def step(self, market_tick):
            # Evaluate both DNAs on this tick
            for label, dna, metrics in [('control', self.control_dna, self.control_metrics), ('variant', self.variant_dna, self.variant_metrics)]:
                indicator_value = market_tick.get(dna.trigger_indicator, None)
                if indicator_value is None:
                    continue
                op = dna.trigger_operator
                threshold = dna.trigger_threshold
                triggered = (op == '<' and indicator_value < threshold) or (op == '>' and indicator_value > threshold)
                if triggered:
                    metrics['activations'] += 1
                    price_now = market_tick.get('price', 0.0)
                    price_prev = market_tick.get('price_prev', price_now)
                    if dna.action_target.startswith('buy'):
                        pnl = 1.0 if price_now > price_prev else -1.0
                    elif dna.action_target.startswith('sell'):
                        pnl = 1.0 if price_now < price_prev else -1.0
                    else:
                        pnl = 0.0
                    metrics['pnl'] += pnl
                    metrics['pnl_history'].append(pnl)
            self.ticks += 1
            if self.ticks >= AB_TEST_TICKS:
                self.finished = True

        def summary(self):
            def sharpe(metrics):
                if not metrics['pnl_history']:
                    return 0.0
                mean = np.mean(metrics['pnl_history'])
                std = np.std(metrics['pnl_history']) + 1e-6
                return mean / std
            return {
                'control': {
                    'pnl': self.control_metrics['pnl'],
                    'activations': self.control_metrics['activations'],
                    'sharpe': sharpe(self.control_metrics)
                },
                'variant': {
                    'pnl': self.variant_metrics['pnl'],
                    'activations': self.variant_metrics['activations'],
                    'sharpe': sharpe(self.variant_metrics)
                },
                'regime': self.regime,
                'ticks': self.ticks
            }

    def start_ab_test(self, candidate_id, control_dna, variant_dna, start_tick, regime):
        self.active_tests[candidate_id] = self.ABTest(candidate_id, control_dna, variant_dna, start_tick, regime)

    def step_all(self, market_tick):
        finished = []
        for cid, abtest in self.active_tests.items():
            abtest.step(market_tick)
            if abtest.finished:
                finished.append(cid)
        return finished

    def get_finished(self):
        return [cid for cid, abtest in self.active_tests.items() if abtest.finished]

    def pop_finished(self):
        finished = self.get_finished()
        results = {cid: self.active_tests.pop(cid).summary() for cid in finished}
        return results

class SystemOrchestrator:
    """
    Minimal v1.0 SystemOrchestrator for Project Crypto Bot Peter.
    Loads configuration, manages LEE and MarketPersonas, and runs evolutionary cycles.
    """
    def __init__(self, config_file_path: str = None, mode: str = 'FULL_V1_2', performance_log_path: str = 'performance_log_FULL_V1_2.csv'):
        self.mode = mode
        self.performance_log_path = performance_log_path
        self.lee_instance = None
        self.available_personas = {}
        self.active_persona_name = None
        self.current_generation = 0
        self.mle_instance = None
        self.ces_instance = None
        self.current_mle_bias = {}
        self.current_ces_vector = {}
        self.priming_generations = 10  # Default, can be overridden by config
        self._load_config(config_file_path)
        # Ensure LEE and MLE use the correct log path
        self.lee_instance.performance_log_path = self.performance_log_path
        self.mle_instance.performance_log_path = self.performance_log_path

    def _load_config(self, config_file_path):
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        # Load LEE parameters
        lee_params = config['lee_params']
        self.lee_instance = LEE(**lee_params)
        # Load MarketPersonas
        personas = config['personas']
        for name, weights in personas.items():
            self.available_personas[name] = MarketPersona(name, weights)
        # Set initial active persona
        self.active_persona_name = config.get('initial_active_persona', list(self.available_personas.keys())[0])
        # Optionally, load seed DNA templates (not implemented for v1.0)
        self.lee_instance.initialize_population()
        self.current_generation = 0
        self.performance_log_path = config.get('performance_log_path', 'performance_log.csv')
        self.mle_instance = MLE_v0_1(self.performance_log_path)
        self.ces_instance = CES_v1_0()
        self.priming_generations = config.get('priming_generations', 10)
        logging.info(f"SystemOrchestrator initialized. Active persona: {self.active_persona_name}")

    def set_active_persona(self, persona_name: str):
        if persona_name in self.available_personas:
            self.active_persona_name = persona_name
            logging.info(f"Active persona set to: {persona_name}")
        else:
            raise ValueError(f"Persona '{persona_name}' not found in available_personas.")

    def run_n_generations(self, num_generations: int, market_data_snapshots: list):
        for gen in range(num_generations):
            phase = 'Priming' if self.current_generation < self.priming_generations else 'Active Feedback'
            print(f"\n=== Generation {self.current_generation+1} | Phase: {phase} ===")
            # Get current market data snapshot
            current_market_data = market_data_snapshots[gen % len(market_data_snapshots)]
            # Always ensure performance_log_path is included in market_data_for_evaluation
            current_market_data = dict(current_market_data)  # Copy to avoid mutating input
            current_market_data['performance_log_path'] = self.performance_log_path
            current_market_data['current_generation'] = self.current_generation + 1
            if self.current_generation < self.priming_generations:
                # Priming phase: neutral MLE bias, CES with None for MLE input
                mle_bias = {'seed_motifs': {}, 'recommended_operator_biases': {}}
                ces_vector = self.ces_instance.calculate_ces_vector(current_market_data, None)
                self.current_mle_bias = mle_bias
                self.current_ces_vector = ces_vector
                print(f"[Priming] CES Vector: {ces_vector}")
                # Enhanced persona selection logic
                v = ces_vector.get('volatility', 0.5)
                t = ces_vector.get('trend', 0.5)
                l = ces_vector.get('liquidity', 0.5)
                if v > 0.6 and t > 0.6:
                    persona_name = 'HUNTER_v1'
                elif v > 0.7 and l < 0.4:
                    persona_name = 'GUARDIAN_v1'
                else:
                    persona_name = 'HUNTER_v1'
                self.set_active_persona(persona_name)
                print(f"[Priming] Persona selected: {self.active_persona_name}")
                persona = self.available_personas[self.active_persona_name]
                self.lee_instance.run_evolutionary_cycle(persona, {**current_market_data, 'ces_vector': ces_vector, 'performance_log_path': self.performance_log_path, 'current_generation': self.current_generation+1}, mle_bias, ces_vector)
                # After evaluation, let MLE learn (but don't use its output yet)
                try:
                    self.mle_instance.analyze_recent_performance()
                except Exception as e:
                    print(f"[Priming] MLE analysis skipped: {e}")
            else:
                # Active feedback phase: use MLE output as input to CES
                try:
                    mle_bias = self.mle_instance.analyze_recent_performance()
                except Exception as e:
                    print(f"[Active] MLE analysis failed: {e}")
                    mle_bias = {'seed_motifs': {}, 'recommended_operator_biases': {}}
                ces_vector = self.ces_instance.calculate_ces_vector(current_market_data, {'pattern_regime_confidence': mle_bias.get('some_mle_confidence_metric', 0.0)})
                self.current_mle_bias = mle_bias
                self.current_ces_vector = ces_vector
                print(f"[Active] CES Vector: {ces_vector}")
                print(f"[Active] MLE Bias: {mle_bias}")
                # Enhanced persona selection logic
                v = ces_vector.get('volatility', 0.5)
                t = ces_vector.get('trend', 0.5)
                l = ces_vector.get('liquidity', 0.5)
                if v > 0.6 and t > 0.6:
                    persona_name = 'HUNTER_v1'
                elif v > 0.7 and l < 0.4:
                    persona_name = 'GUARDIAN_v1'
                else:
                    persona_name = 'HUNTER_v1'
                self.set_active_persona(persona_name)
                print(f"[Active] Persona selected: {self.active_persona_name}")
                persona = self.available_personas[self.active_persona_name]
                self.lee_instance.run_evolutionary_cycle(persona, {**current_market_data, 'ces_vector': ces_vector, 'performance_log_path': self.performance_log_path, 'current_generation': self.current_generation+1}, mle_bias, ces_vector)
            self.current_generation += 1
            print(f"Generation {self.current_generation} complete. Active persona: {self.active_persona_name}")

    def harmonize_influences(self, base_decision_parameters: dict, system_influences: dict) -> dict:
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
        logger.debug(f"ORCHESTRATOR_HARMONIZE_INPUT|base_params={base_decision_parameters}|influences={system_influences}")
        final_params = base_decision_parameters.copy()
        # FSM wake threshold integration
        if 'fsm_wake_threshold' in final_params:
            base_wake = final_params['fsm_wake_threshold']
            modified_wake = base_wake
            if 'persona_wake_factor' in system_influences:
                modified_wake *= system_influences['persona_wake_factor']
            if 'env_score_wake_factor' in system_influences:
                modified_wake *= system_influences['env_score_wake_factor']
            # Add other influences as needed
            min_thr = self.config.getfloat('OrchestratorClamps', 'min_fsm_wake_threshold', fallback=0.00001)
            max_thr = self.config.getfloat('OrchestratorClamps', 'max_fsm_wake_threshold', fallback=0.01)
            final_params['fsm_wake_threshold'] = np.clip(modified_wake, min_thr, max_thr)
        # L1 action confidence integration
        if 'l1_action_confidence' in final_params and 'opportunistic_confidence_boost' in system_influences:
            base_conf = final_params['l1_action_confidence']
            boost = system_influences['opportunistic_confidence_boost']
            modified_conf = base_conf + boost
            final_params['l1_action_confidence'] = np.clip(modified_conf, 0.0, 1.0)
        # Anomaly sequence suggestion (MVP: log only)
        if 'anomaly_sequence_suggestion' in system_influences:
            logger.info(f"ORCHESTRATOR_SEQUENCE_SUGGESTION|suggestion={system_influences['anomaly_sequence_suggestion']}")
        # Conflict resolution (stub)
        if self.conflict_resolver.detect_conflicts(final_params, system_influences):
            final_params = self.conflict_resolver.resolve(final_params, system_influences)
        logger.info(f"ORCHESTRATOR_HARMONIZED_OUTPUT|final_params={final_params}")
        return final_params

    def promote_to_candidate_slot(self, graduated_entry):
        if len(self.candidate_slots) < MAX_CANDIDATE_SLOTS:
            slot = CandidateStrategySlot(graduated_entry)
            self.candidate_slots.append(slot)
            self.logger.info(f"SO: Promoted DNA to candidate slot: {slot.dna} | ParentID={slot.dna.parent_id}")
            self.log_event_snapshot('GRADUATION', slot.dna, graduated_entry)
        else:
            # Replace worst performing candidate
            worst_idx = min(range(len(self.candidate_slots)), key=lambda i: self.candidate_slots[i].pnl_candidate)
            self.logger.info(f"SO: Candidate slots full. Demoting worst: {self.candidate_slots[worst_idx].dna}")
            self.log_event_snapshot('DEMOTION', self.candidate_slots[worst_idx].dna, self.candidate_slots[worst_idx])
            self.candidate_slots.pop(worst_idx)
            slot = CandidateStrategySlot(graduated_entry)
            self.candidate_slots.append(slot)
            self.logger.info(f"SO: Promoted DNA to candidate slot: {slot.dna} | ParentID={slot.dna.parent_id}")
            self.log_event_snapshot('GRADUATION', slot.dna, graduated_entry)

    def evaluate_candidates_per_tick(self, market_tick, recent_market_data=None):
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