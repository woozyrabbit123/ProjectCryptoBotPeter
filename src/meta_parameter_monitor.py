import time
from datetime import datetime
from src.data_logger import log_event

def check_settings_dict(settings_dict, required_keys, dict_name):
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        raise RuntimeError(f"CRITICAL: Missing keys in {dict_name}: {missing}")

class MetaParameterMonitor:
    def __init__(self, settings, lee_settings, feral_calibrator_settings):
        # Defensive check for meta-thermostat keys
        required_keys = [
            'ENABLE_META_SELF_TUNING', 'META_PARAM_WINDOW_SECONDS', 'META_PARAM_MIN_GENERATIONS',
            'META_PARAM_MIN_AB_TESTS', 'META_PARAM_LOW_SURVIVAL_THRESHOLD',
            'META_PARAM_HIGH_EXPLORER_RATIO_THRESHOLD', 'META_PARAM_GTS_ADJUST_PCT',
            'META_PARAM_GTS_MIN', 'META_PARAM_GTS_MAX', 'META_PARAM_GTS_COOLDOWN_WINDOWS',
            'META_PARAM_LOW_AB_ADOPTION_THRESHOLD', 'META_PARAM_HIGH_AB_ADOPTION_THRESHOLD',
            'META_PARAM_AB_ADOPT_ADJUST_PCT', 'META_PARAM_AB_ADOPT_MIN', 'META_PARAM_AB_ADOPT_MAX',
            'META_PARAM_AB_ADOPT_COOLDOWN_WINDOWS'
        ]
        check_settings_dict(settings.__dict__ if hasattr(settings, '__dict__') else settings, required_keys, 'META_PARAM_SETTINGS')
        self.settings = settings
        self.lee_settings = lee_settings
        self.feral_calibrator_settings = feral_calibrator_settings
        self.enabled = getattr(settings, 'ENABLE_META_SELF_TUNING', False)
        # Event history
        self.mvl_generations = []  # (timestamp, dna_id)
        self.mvl_survivals = []    # (timestamp, dna_id)
        self.graduations = []      # (timestamp, dna_id)
        self.ab_tests = []         # (timestamp, ab_test_id)
        self.ab_adoptions = []     # (timestamp, ab_test_id)
        # In-memory overrides
        self.generation_trigger_sensitivity = lee_settings['GENERATION_TRIGGER_SENSITIVITY']
        self.ab_test_adoption_sharpe_uplift_min = feral_calibrator_settings['AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN']
        # Cooldown trackers
        self.cooldowns = {
            'GENERATION_TRIGGER_SENSITIVITY': 0,
            'AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN': 0,
        }
        self.last_adjustment_window = {
            'GENERATION_TRIGGER_SENSITIVITY': -1000,
            'AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN': -1000,
        }
        self.window_counter = 0

    def notify_mvl_generation(self, dna_id):
        self.mvl_generations.append((time.time(), dna_id))

    def notify_mvl_survival(self, dna_id):
        self.mvl_survivals.append((time.time(), dna_id))

    def notify_graduation(self, dna_id):
        self.graduations.append((time.time(), dna_id))

    def notify_ab_test(self, ab_test_id):
        self.ab_tests.append((time.time(), ab_test_id))

    def notify_ab_adoption(self, ab_test_id):
        self.ab_adoptions.append((time.time(), ab_test_id))

    def get_windowed_events(self, events, window_size, now=None):
        now = now or time.time()
        return [e for e in events if now - e[0] <= window_size]

    def evaluate_and_adjust(self):
        if not self.enabled:
            return
        now = time.time()
        # Config
        window_sec = getattr(self.settings, 'META_PARAM_WINDOW_SECONDS', 1800)
        min_gens = getattr(self.settings, 'META_PARAM_MIN_GENERATIONS', 20)
        min_ab_tests = getattr(self.settings, 'META_PARAM_MIN_AB_TESTS', 5)
        # --- Calculate metrics ---
        gens = self.get_windowed_events(self.mvl_generations, window_sec, now)
        survs = self.get_windowed_events(self.mvl_survivals, window_sec, now)
        grads = self.get_windowed_events(self.graduations, window_sec, now)
        ab_tests = self.get_windowed_events(self.ab_tests, window_sec, now)
        ab_adopts = self.get_windowed_events(self.ab_adoptions, window_sec, now)
        mvl_survival_rate = len(survs) / max(1, len(gens))
        explorer_ratio = len(gens) / max(1, len(grads))
        ab_test_adoption_rate = len(ab_adopts) / max(1, len(ab_tests))
        # --- Adjustment logic ---
        self.window_counter += 1
        # GENERATION_TRIGGER_SENSITIVITY
        cooldown_gts = self.cooldowns['GENERATION_TRIGGER_SENSITIVITY']
        if self.window_counter > cooldown_gts:
            low_surv = getattr(self.settings, 'META_PARAM_LOW_SURVIVAL_THRESHOLD', 0.08)
            high_explorer = getattr(self.settings, 'META_PARAM_HIGH_EXPLORER_RATIO_THRESHOLD', 20)
            adjust_pct = getattr(self.settings, 'META_PARAM_GTS_ADJUST_PCT', 0.05)
            min_gts = getattr(self.settings, 'META_PARAM_GTS_MIN', 0.01)
            max_gts = getattr(self.settings, 'META_PARAM_GTS_MAX', 1.0)
            cooldown_windows = getattr(self.settings, 'META_PARAM_GTS_COOLDOWN_WINDOWS', 3)
            # Check sustained low survival and high explorer ratio
            if mvl_survival_rate < low_surv and explorer_ratio > high_explorer:
                old_val = self.generation_trigger_sensitivity
                new_val = min(max_gts, old_val * (1 + adjust_pct))
                if new_val != old_val:
                    self.generation_trigger_sensitivity = new_val
                    self.cooldowns['GENERATION_TRIGGER_SENSITIVITY'] = self.window_counter + cooldown_windows
                    log_event('META_PARAM_SELF_ADJUSTED', {
                        'timestamp': datetime.now().isoformat(),
                        'parameter_name': 'LEE_GENERATION_TRIGGER_SENSITIVITY',
                        'old_value': old_val,
                        'new_value': new_val,
                        'triggering_metric_name': 'mvl_survival_rate',
                        'triggering_metric_value': mvl_survival_rate,
                        'threshold_for_trigger': low_surv,
                        'adjustment_reason': 'Slowing DNA generation due to low MVL survival and high explorer ratio.',
                        'cooldown_initiated_until_window': self.cooldowns['GENERATION_TRIGGER_SENSITIVITY'],
                    })
        # AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN
        cooldown_ab = self.cooldowns['AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN']
        if self.window_counter > cooldown_ab:
            low_ab = getattr(self.settings, 'META_PARAM_LOW_AB_ADOPTION_THRESHOLD', 0.05)
            high_ab = getattr(self.settings, 'META_PARAM_HIGH_AB_ADOPTION_THRESHOLD', 0.5)
            adjust_pct = getattr(self.settings, 'META_PARAM_AB_ADOPT_ADJUST_PCT', 0.1)
            min_ab = getattr(self.settings, 'META_PARAM_AB_ADOPT_MIN', 0.01)
            max_ab = getattr(self.settings, 'META_PARAM_AB_ADOPT_MAX', 1.0)
            cooldown_windows = getattr(self.settings, 'META_PARAM_AB_ADOPT_COOLDOWN_WINDOWS', 3)
            # Low adoption rate
            if ab_test_adoption_rate < low_ab:
                old_val = self.ab_test_adoption_sharpe_uplift_min
                new_val = max(min_ab, old_val * (1 - adjust_pct))
                if new_val != old_val:
                    self.ab_test_adoption_sharpe_uplift_min = new_val
                    self.cooldowns['AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN'] = self.window_counter + cooldown_windows
                    log_event('META_PARAM_SELF_ADJUSTED', {
                        'timestamp': datetime.now().isoformat(),
                        'parameter_name': 'FERAL_CALIBRATOR_AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN',
                        'old_value': old_val,
                        'new_value': new_val,
                        'triggering_metric_name': 'ab_test_adoption_rate',
                        'triggering_metric_value': ab_test_adoption_rate,
                        'threshold_for_trigger': low_ab,
                        'adjustment_reason': 'Lowering A/B adoption threshold due to low adoption rate.',
                        'cooldown_initiated_until_window': self.cooldowns['AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN'],
                    })
            # High adoption rate
            elif ab_test_adoption_rate > high_ab:
                old_val = self.ab_test_adoption_sharpe_uplift_min
                new_val = min(max_ab, old_val * (1 + adjust_pct))
                if new_val != old_val:
                    self.ab_test_adoption_sharpe_uplift_min = new_val
                    self.cooldowns['AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN'] = self.window_counter + cooldown_windows
                    log_event('META_PARAM_SELF_ADJUSTED', {
                        'timestamp': datetime.now().isoformat(),
                        'parameter_name': 'FERAL_CALIBRATOR_AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN',
                        'old_value': old_val,
                        'new_value': new_val,
                        'triggering_metric_name': 'ab_test_adoption_rate',
                        'triggering_metric_value': ab_test_adoption_rate,
                        'threshold_for_trigger': high_ab,
                        'adjustment_reason': 'Increasing A/B adoption threshold due to high adoption rate.',
                        'cooldown_initiated_until_window': self.cooldowns['AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN'],
                    }) 