from src.system_orchestrator import SystemOrchestrator

def run_v1_core_demo():
    print("Starting Project Crypto Bot Peter v1.0 Core Demo...")

    # Simple mock market data for demonstration
    mock_market_data = [
        {'timestamp': 1, 'prices_ohlcv': {}, 'indicators': {'metric_for_profit': 10, 'metric_for_trades': 1}},
        {'timestamp': 2, 'prices_ohlcv': {}, 'indicators': {'metric_for_profit': 12, 'metric_for_trades': 2}},
        {'timestamp': 3, 'prices_ohlcv': {}, 'indicators': {'metric_for_profit': 11, 'metric_for_trades': 1}},
        {'timestamp': 4, 'prices_ohlcv': {}, 'indicators': {'metric_for_profit': 13, 'metric_for_trades': 3}},
    ]

    config_file = "config_v1.json"
    orchestrator = SystemOrchestrator(config_file_path=config_file)

    num_demo_generations = 10
    print(f"Running evolution with active persona: {orchestrator.active_persona_name} for {num_demo_generations} generations...")
    orchestrator.run_n_generations(num_demo_generations, mock_market_data)

    print(f"\nEvolution complete after {orchestrator.current_generation} generations.")

    # Print basic stats from the final LEE population
    if orchestrator.lee_instance.population:
        fittest_dna = max(
            orchestrator.lee_instance.population,
            key=lambda dna: dna.fitness if hasattr(dna, 'fitness') else -float('inf')
        )
        if hasattr(fittest_dna, 'fitness'):
            print(f"Fittest DNA in final population - Fitness: {fittest_dna.fitness:.4f}")
            # If to_string_representation is implemented:
            # print(f"Structure: {fittest_dna.to_string_representation()}")
        else:
            print("Final population exists, but fitness not evaluated on all individuals for this print.")

if __name__ == "__main__":
    run_v1_core_demo() 