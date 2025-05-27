# settings.py
"""
Centralized configuration for Project Crypto Bot Peter.
All key parameters, thresholds, and tunables are grouped by module for easy management.
"""

# === LEE (Logic Evolution Engine) Settings ===
LEE_SETTINGS = {
    'MUTATION_STRENGTH': 0.1,  # Default mutation strength for DNA
    'CYCLE_GENERATE_INTERVAL': 3,  # Generate new DNA every N cycles
    'DUMMY_MARKET_DATA_SIZE': 100,  # Number of ticks for dummy data
    'MVL_BUFFER_SIZE': 30,  # Buffer size for micro-sims
    'GENERATION_TRIGGER_SENSITIVITY': 5,  # Number of cycles between DNA generations (default)
}

# === EPM (Experimental Pool Manager) Settings ===
EPM_SETTINGS = {
    'MAX_POOL_SIZE': 10,
    'MAX_POOL_LIFE': 1000,  # ticks
    'MIN_PERFORMANCE_THRESHOLD': -3.0,
    'MIN_POOL_SIZE': 3,
    'GRADUATION_MIN_SHARPE_POOL': 0.5,
    'GRADUATION_MIN_PNL_POOL': 3.0,
    'GRADUATION_MIN_TICKS_IN_POOL': 100,
    'GRADUATION_MIN_ACTIVATIONS_POOL': 3,
    'GRADUATION_EPSILON': 1e-6,
}

# === System Orchestrator (SO) Settings ===
SO_SETTINGS = {
    'MAX_CANDIDATE_SLOTS': 2,
    'CANDIDATE_VIRTUAL_CAPITAL': 10.0,
    'CANDIDATE_MIN_SHARPE': 0.3,
    'CANDIDATE_MIN_PNL': 5.0,
    'CANDIDATE_MIN_TICKS': 100,
    'CANDIDATE_MIN_ACTIVATIONS': 3,
    'CANDIDATE_EPSILON': 1e-6,
    'CANDIDATE_MAX_LIFE': 2000,  # ticks
    'CANDIDATE_SHARPE_DROP_TRIGGER': 0.3,  # 30% drop triggers self-calibration
    'AB_TEST_TICKS': 200,  # A/B test period
    'FITNESS_WEIGHTS': {'sharpe': 0.4, 'regime_consistency': 0.3, 'diversification': 0.3},
    'AB_SHARPE_THRESHOLD': 0.10,  # 10% Sharpe improvement
    'AB_PNL_THRESHOLD': 0.05,     # 5% PnL improvement
    'INFLUENCE_MULTIPLIER_MAX': 1.5,
    'INFLUENCE_MULTIPLIER_MIN': 0.5,
    'INFLUENCE_MULTIPLIER_STEP_UP': 0.1,
    'INFLUENCE_MULTIPLIER_STEP_DOWN': 0.1,
    'INFLUENCE_SCORE_HIGH_PCT': 0.75,  # Top 25% get boost
    'INFLUENCE_SCORE_LOW_PCT': 0.25,   # Bottom 25% get reduced
    'COOLDOWN_TICKS_AFTER_TWEAK': 750,
    'KPI_DASHBOARD_INTERVAL': 10,  # Log KPI dashboard every N cycles
}

# === Feral Calibrator / Adaptive Behavior Settings ===
FERAL_CALIBRATOR_SETTINGS = {
    # Add Feral Calibrator-specific parameters here as needed
    'AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN': 0.10,  # Minimum Sharpe uplift for A/B adoption
    # Example:
    # 'SOME_THRESHOLD': 0.5,
}

# === Logging Settings ===
LOGGING_SETTINGS = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s %(levelname)s %(message)s',
}

# === General/Other Settings ===
GENERAL_SETTINGS = {
    # Add any other global settings here
    'STRUCTURED_DATA_LOG_FILE': 'project_peter_run_data.jsonl',
}

# === Meta-Thermostat v0.1 (MetaParameterMonitor) ===
ENABLE_META_SELF_TUNING = False  # Master switch for meta-parameter self-tuning
META_PARAM_WINDOW_SECONDS = 1800  # Sliding window size in seconds for metric calculations
META_PARAM_MIN_GENERATIONS = 20  # Minimum DNA generations in window to trigger evaluation
META_PARAM_MIN_AB_TESTS = 5  # Minimum A/B tests in window to trigger evaluation
META_PARAM_LOW_SURVIVAL_THRESHOLD = 0.08  # If MVL survival rate < this, triggers adjustment
META_PARAM_HIGH_EXPLORER_RATIO_THRESHOLD = 20  # If explorer ratio > this, triggers adjustment
META_PARAM_GTS_ADJUST_PCT = 0.05  # % to adjust GENERATION_TRIGGER_SENSITIVITY per change
META_PARAM_GTS_MIN = 0.01  # Min bound for GENERATION_TRIGGER_SENSITIVITY
META_PARAM_GTS_MAX = 1.0   # Max bound for GENERATION_TRIGGER_SENSITIVITY
META_PARAM_GTS_COOLDOWN_WINDOWS = 3  # Cooldown windows after GTS adjustment
META_PARAM_LOW_AB_ADOPTION_THRESHOLD = 0.05  # If A/B adoption rate < this, lower threshold
META_PARAM_HIGH_AB_ADOPTION_THRESHOLD = 0.5  # If A/B adoption rate > this, raise threshold
META_PARAM_AB_ADOPT_ADJUST_PCT = 0.1  # % to adjust AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN per change
META_PARAM_AB_ADOPT_MIN = 0.01  # Min bound for AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN
META_PARAM_AB_ADOPT_MAX = 1.0   # Max bound for AB_TEST_ADOPTION_SHARPE_UPLIFT_MIN
META_PARAM_AB_ADOPT_COOLDOWN_WINDOWS = 3  # Cooldown windows after AB adoption adjustment 