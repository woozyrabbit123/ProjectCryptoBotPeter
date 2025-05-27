# Project Crypto Bot Peter

## Overview
Project Crypto Bot Peter is an experimental platform for developing and testing automated cryptocurrency trading strategies using evolutionary algorithms and machine learning techniques.

## Core Components
*   **`src/system_orchestrator.py`**: Central coordinator managing modules, configurations, strategy lifecycles, and overall system flow.
*   **`src/lee.py` (Logic Evolution Engine)**: Evolves trading strategy logic (`LogicDNA`) through simulated evolutionary cycles.
*   **`src/mle_engine.py` (Meta-Learning Engine)**: Analyzes performance logs to identify successful patterns and provide feedback to the evolution process.
*   **`src/logic_dna.py`**: Defines the structure and representation of trading strategies (decision trees/rules).
*   **`src/ces_module.py` (Contextual Environment Scorer)**: Assesses current market conditions (e.g., volatility, trend) to provide context.
*   **`src/data_handling.py`**: Utilities for loading and processing market data.
*   **`src/utils/logging_utils.py`**: Provides centralized configuration for application-wide logging.

## Running the Bot
The main entry point for running the system is `main_v1_1.py`.

Example:
```bash
python main_v1_1.py --mode FULL_V1_2
```
*   `--mode`: Specifies the operational mode of the bot (e.g., `FULL_V1_2` for full operation, `LEE_ONLY_V1_1` for testing LEE in isolation). Consult `main_v1_1.py` for all available modes.

## Configuration
The project uses two main configuration files:

*   **`config.json`**: Used by `SystemOrchestrator` for core operational parameters, including strategy evolution settings, market persona definitions, paths for RNG state management, and parameters for various bot components.
*   **`config.ini`**: Used by `src/utils/logging_utils.py` to configure application-wide logging behavior, such as log level, file output path, and log formats.

## Reproducibility
To ensure experiments can be reproduced, the system can save and load the state of its random number generators (RNGs). This is managed by `SystemOrchestrator`.

*   **Configuration**: This feature is controlled by `rng_state_load_path` (to load a previously saved state at startup) and `rng_state_save_path` (to save the current state at exit or interruption) in the `config.json` file.
*   **Coverage**: Both Python's built-in `random` module and `numpy.random` module states are managed.
*   **Security Note**: Important: RNG state files are typically saved using `pickle`. Only load RNG state files from trusted sources, as malicious pickle files can lead to arbitrary code execution.
```
