import pandas as pd
from src.data_handling import load_ohlcv_csv, calculate_indicators
from src.logic_dna import LogicDNA_v1, ConditionNode, ActionNode, CompositeNode
from src.lee import LEE

# 1. Load mock OHLCV data
ohlcv = load_ohlcv_csv('data/mock_ohlcv_data.csv')

# 2. Calculate indicators
indicators = calculate_indicators(ohlcv)

# 3. Create a simple LogicDNA_v1 individual
# Example: IF SMA_10 > 105 THEN BUY 100%
condition = ConditionNode(
    indicator_id='SMA',
    comparison_operator='GREATER_THAN',
    threshold_value=105.0,
    lookback_period_1=10
)
action = ActionNode(action_type='BUY', size_factor=1.0)
root = CompositeNode('AND', condition, action)
dna = LogicDNA_v1(root_node=root)

# 4. Define backtest config
default_backtest_config = {
    'initial_capital': 10000.0,
    'transaction_cost_pct': 0.001
}

# 5. Instantiate a LEE (just to use its backtest method)
lee = LEE(
    population_size=1,
    mutation_rate_parametric=0.1,
    mutation_rate_structural=0.2,
    crossover_rate=0.3,
    elitism_percentage=0.2,
    random_injection_percentage=0.1,
    max_depth=4,
    max_nodes=9,
    complexity_weights={'nodes': 0.5, 'depth': 0.5}
)

# 6. Run the backtest for the single DNA
metrics = lee._run_backtest_for_dna(
    dna,
    historical_ohlcv=ohlcv,
    precomputed_indicators=indicators,
    backtest_config=default_backtest_config
)

# 7. Print the results
print("\n=== Backtest Results for Single LogicDNA_v1 Individual ===")
for k, v in metrics.items():
    if k != 'trade_log':
        print(f"{k}: {v}")
print("Trade Log (first 5 shown):")
for trade in metrics['trade_log'][:5]:
    print(trade)
if len(metrics['trade_log']) > 5:
    print(f"... ({len(metrics['trade_log'])} total trades)") 