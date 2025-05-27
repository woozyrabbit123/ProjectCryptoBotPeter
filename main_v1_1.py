from src.data_handling import load_ohlcv_csv, calculate_indicators
from src.system_orchestrator import SystemOrchestrator
import os
import argparse

def run_v1_1_core_demo():
    print("DEBUG: Entering run_v1_1_core_demo function")
    parser = argparse.ArgumentParser(description="Project Crypto Bot Peter v1.2 Validation Demo")
    parser.add_argument('--mode', type=str, default='FULL_V1_2',
                        choices=['LEE_ONLY', 'LEE_CES', 'LEE_MLE', 'FULL_V1_2', 'LEE_MLE_RANDOM'],
                        help='Operational mode for ablation/validation study')
    args = parser.parse_args()
    mode = args.mode
    print(f"DEBUG: Mode selected: {mode}")
    print(f"\n[INFO] Running in mode: {mode}")
    print("DEBUG: About to load historical OHLCV data")
    # Load historical OHLCV data and calculate indicators
    ohlcv = load_ohlcv_csv('data/mock_ohlcv_data.csv')
    indicators = calculate_indicators(ohlcv)
    # Define backtest config
    backtest_config = {
        'initial_capital': 10000.0,
        'transaction_cost_pct': 0.001
    }
    # Performance log path
    performance_log_path = f'performance_log_{mode}.csv'
    if os.path.exists(performance_log_path):
        os.remove(performance_log_path)  # Start fresh for demo
    print("DEBUG: About to instantiate SystemOrchestrator")
    # Instantiate orchestrator with mode
    config_file = "config_v1.json"
    orchestrator = SystemOrchestrator(config_file_path=config_file, mode=mode, performance_log_path=performance_log_path)
    num_demo_generations = 15  # Demo both priming and feedback phases
    print(f"Running evolution with active persona: {orchestrator.active_persona_name} for {num_demo_generations} generations...")
    # Prepare a list of market_data_snapshots (repeat the same for demo)
    market_data_snapshot = {
        'ohlcv': ohlcv,
        'indicators': indicators,
        'backtest_config': backtest_config,
        'active_persona_name': orchestrator.active_persona_name
    }
    market_data_snapshots = [market_data_snapshot for _ in range(num_demo_generations)]
    print("DEBUG: About to call orchestrator.run_n_generations")
    orchestrator.run_n_generations(num_demo_generations, market_data_snapshots)
    print(f"\nEvolution complete after {orchestrator.current_generation} generations.")
    print(f"Performance log written to: {performance_log_path}")

if __name__ == "__main__":
    run_v1_1_core_demo() 