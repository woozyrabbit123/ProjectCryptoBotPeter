import pytest
from src.lee import LEE
from src.logic_dna import LogicDNA_v1, ConditionNode, ActionNode, CompositeNode, SequenceNode
from src.market_persona import MarketPersona
import random
import pandas as pd
import numpy as np
import os

def make_seed_dna():
    # Simple seed: Composite(AND, Condition, Action)
    cond = ConditionNode("SMA", "GREATER_THAN", 10.0, 5)
    act = ActionNode("BUY", 0.5)
    root = CompositeNode("AND", cond, act)
    return LogicDNA_v1(root_node=root)

def make_mock_market_data_for_evaluation():
    # Create a small mock ohlcv DataFrame
    ohlcv = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'open': np.linspace(100, 110, 10),
        'high': np.linspace(101, 111, 10),
        'low': np.linspace(99, 109, 10),
        'close': np.linspace(100, 110, 10),
        'volume': np.linspace(1000, 2000, 10)
    })
    # Create a small mock indicators DataFrame
    indicators = pd.DataFrame({
        'SMA_10': np.linspace(100, 110, 10),
        'EMA_10': np.linspace(100, 110, 10),
        'RSI_14': np.linspace(30, 70, 10),
        'VOLATILITY_20': np.linspace(0.01, 0.05, 10)
    })
    backtest_config = {'initial_capital': 10000, 'transaction_cost_pct': 0.001}
    return {
        'ohlcv': ohlcv,
        'indicators': indicators,
        'backtest_config': backtest_config,
        'active_persona_name': 'TEST_PERSONA',
        'ces_vector': {},
        'performance_log_path': 'test_performance_log.csv',
        'current_generation': 1
    }

def test_lee_initialization():
    lee = LEE(
        population_size=10,
        mutation_rate_parametric=0.1,
        mutation_rate_structural=0.2,
        crossover_rate=0.3,
        elitism_percentage=0.1,
        random_injection_percentage=0.1,
        max_depth=4,
        max_nodes=9,
        complexity_weights={'nodes': 0.5, 'depth': 0.5}
    )
    assert lee.population_size == 10
    assert lee.max_depth == 4
    assert lee.complexity_weights['nodes'] == 0.5
    assert isinstance(lee.population, list)

def test_initialize_population_random_only():
    lee = LEE(8, 0.1, 0.2, 0.3, 0.1, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    lee.initialize_population()
    assert len(lee.population) == 8
    for dna in lee.population:
        assert dna.is_valid(lee.max_depth, lee.max_nodes)

def test_initialize_population_with_seeds():
    seed = make_seed_dna()
    lee = LEE(10, 0.1, 0.2, 0.3, 0.1, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    lee.initialize_population(seed_dna_templates=[seed])
    assert len(lee.population) == 10
    # At least 2 (20%) should be based on the seed
    seed_count = 0
    for dna in lee.population:
        if (isinstance(dna.root_node, CompositeNode) and
            isinstance(dna.root_node.child1, ConditionNode) and
            isinstance(dna.root_node.child2, ActionNode)):
            seed_count += 1
        assert dna.is_valid(lee.max_depth, lee.max_nodes)
    assert seed_count >= 2

def make_mock_persona():
    # Simple persona: profit focus, small complexity penalty
    return MarketPersona('TestPersona', {'metric_profit': 1.0, 'complexity_penalty_weight': 0.1})

def make_diverse_population():
    # Returns a list of valid, diverse LogicDNA_v1 trees
    pop = []
    for i in range(3):
        cond = ConditionNode("SMA", "GREATER_THAN", 10.0 + i, 5 + i)
        act = ActionNode("BUY", 0.5 + 0.1 * i)
        root = CompositeNode("AND", cond, act)
        pop.append(LogicDNA_v1(root_node=root))
    return pop

def test_evaluate_population():
    random.seed(100)
    lee = LEE(3, 0.1, 0.2, 0.3, 0.1, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    lee.population = make_diverse_population()
    persona = make_mock_persona()
    market_data_for_evaluation = make_mock_market_data_for_evaluation()
    # Remove log file if exists
    if os.path.exists(market_data_for_evaluation['performance_log_path']):
        os.remove(market_data_for_evaluation['performance_log_path'])
    lee._evaluate_population(persona, market_data_for_evaluation)
    for dna in lee.population:
        assert hasattr(dna, 'fitness')
        assert isinstance(dna.fitness, float)

def test_select_parents_elitism_and_tournament():
    random.seed(101)
    lee = LEE(6, 0.1, 0.2, 0.3, 0.5, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    lee.population = make_diverse_population() * 2  # 6 individuals
    # Assign mock fitness
    for i, dna in enumerate(lee.population):
        dna.fitness = float(i)
    elites, parents = lee._select_parents()
    assert len(elites) == 3  # 50% of 6
    assert all(isinstance(e, LogicDNA_v1) for e in elites)
    assert len(parents) == 6
    # Elites should be the top fitness individuals
    elite_fitnesses = [e.fitness for e in elites]
    assert elite_fitnesses == sorted(elite_fitnesses, reverse=True)

def test_parametric_mutation_changes_parameters():
    random.seed(102)
    lee = LEE(1, 0.1, 0.2, 0.3, 0.1, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    dna = make_seed_dna()
    dna_copy = dna.copy()
    mutated = lee._parametric_mutation(dna_copy)
    # Original unchanged
    assert dna.root_node.child1.threshold_value == 10.0
    # Mutated may differ
    changed = (
        mutated.root_node.child1.threshold_value != 10.0 or
        mutated.root_node.child2.size_factor != 0.5
    )
    assert changed
    assert mutated.is_valid(lee.max_depth, lee.max_nodes)

def make_sample_dna_for_mutation_test():
    # Create a tree that's not too simple, e.g., depth 2 or 3
    c1 = ConditionNode("SMA", "GREATER_THAN", 10.0, 10)
    a1 = ActionNode("BUY", 0.5)
    s1 = SequenceNode(c1, a1)
    c2 = ConditionNode("RSI", "LESS_THAN", 30.0, 14)
    a2 = ActionNode("SELL", 0.3)
    root = CompositeNode("AND", s1, SequenceNode(c2, a2))
    return LogicDNA_v1(root_node=root)

def test_structural_mutation_changes_structure():
    random.seed(42)  # For test reproducibility
    lee_params = {
        'population_size': 1, 'mutation_rate_parametric': 0.1,
        'mutation_rate_structural': 1.0,  # Ensure structural mutation happens
        'crossover_rate': 0.1, 'elitism_percentage': 0.1,
        'random_injection_percentage': 0.1, 'max_depth': 4, 'max_nodes': 9,
        'complexity_weights': {'nodes': 0.5, 'depth': 0.5}
    }
    lee = LEE(**lee_params)

    original_dna = make_sample_dna_for_mutation_test()
    original_complexity = original_dna.calculate_complexity()
    original_root_type = type(original_dna.root_node)

    changed_structure = False
    attempts = 20  # Try multiple times

    for _ in range(attempts):
        mutated_dna = original_dna.copy()  # Start with a fresh copy for each mutation attempt
        mutated_dna = lee._structural_mutation(mutated_dna)  # Apply mutation

        # Validate the mutated DNA first
        assert mutated_dna.is_valid(lee.max_depth, lee.max_nodes), "Mutated DNA is not valid"

        mutated_complexity = mutated_dna.calculate_complexity()
        mutated_root_type = type(mutated_dna.root_node)

        if (original_complexity[0] != mutated_complexity[0] or \
            original_root_type != mutated_root_type):
            changed_structure = True
            break

    assert changed_structure, f"Structural mutation did not alter node count or root type after {attempts} attempts on a sample DNA."

def test_crossover_trees_produces_offspring():
    random.seed(104)
    lee = LEE(2, 0.1, 0.2, 1.0, 0.1, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    parent1 = make_seed_dna()
    parent2 = make_seed_dna()
    # Make parent2 different
    parent2.root_node.child1.threshold_value = 99.0
    child1, child2 = lee._crossover_trees(parent1, parent2)
    # Parents unchanged
    assert parent1.root_node.child1.threshold_value == 10.0
    assert parent2.root_node.child1.threshold_value == 99.0
    # Offspring are valid and not identical to both parents
    assert child1.is_valid(lee.max_depth, lee.max_nodes)
    assert child2.is_valid(lee.max_depth, lee.max_nodes)
    # At least one child should differ from at least one parent
    assert (
        child1.root_node.child1.threshold_value != 10.0 or
        child2.root_node.child1.threshold_value != 99.0
    )

def test_reproduce_generates_offspring():
    random.seed(105)
    lee = LEE(4, 0.1, 0.2, 1.0, 0.1, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    parents = [make_seed_dna() for _ in range(4)]
    offspring = lee._reproduce(parents, 4)
    assert len(offspring) == 4
    for child in offspring:
        assert isinstance(child, LogicDNA_v1)
        assert child.is_valid(lee.max_depth, lee.max_nodes)

def test_run_evolutionary_cycle_updates_population():
    random.seed(106)
    lee = LEE(10, 0.1, 0.2, 1.0, 0.2, 0.2, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    lee.initialize_population()
    persona = make_mock_persona()
    market_data_for_evaluation = make_mock_market_data_for_evaluation()
    # Remove log file if exists
    if os.path.exists(market_data_for_evaluation['performance_log_path']):
        os.remove(market_data_for_evaluation['performance_log_path'])
    old_population = [dna.copy() for dna in lee.population]
    lee._evaluate_population(persona, market_data_for_evaluation)
    fitnesses = [getattr(dna, 'fitness', 0.0) for dna in lee.population]
    best = max(fitnesses) if fitnesses else None
    avg = sum(fitnesses)/len(fitnesses) if fitnesses else None
    # Evolve population
    elites, parents = lee._select_parents()
    num_random_injections = int(lee.random_injection_percentage * lee.population_size)
    num_offspring_needed = lee.population_size - len(elites) - num_random_injections
    offspring = lee._reproduce(parents, num_offspring_needed)
    random_new = [lee._create_random_valid_tree() for _ in range(num_random_injections)]
    lee.population = elites + offspring + random_new
    assert len(lee.population) == 10
    for dna in lee.population:
        assert dna.is_valid(lee.max_depth, lee.max_nodes)
    # Population should have changed
    changed = any(
        (dna1.root_node.to_string() != dna2.root_node.to_string())
        for dna1, dna2 in zip(old_population, lee.population)
    )
    assert changed

def test_motif_influenced_random_tree_generation():
    from src.lee import LEE
    from src.logic_dna import ConditionNode
    random.seed(123)
    lee = LEE(1, 0.1, 0.2, 0.3, 0.1, 0.1, 4, 9, {'nodes': 0.5, 'depth': 0.5})
    # Strong motif bias for Indicator_RSI_14_Used
    mle_bias = {'seed_motifs': {'Indicator_RSI_14_Used': 100}, 'recommended_operator_biases': {}}
    count_rsi = 0
    trials = 120
    for _ in range(trials):
        tree = lee._create_random_valid_tree(mle_bias=mle_bias)
        # Check if root is a ConditionNode for RSI_14
        if isinstance(tree.root_node, ConditionNode) and tree.root_node.indicator_id == 'RSI' and tree.root_node.lookback_period_1 == 14:
            count_rsi += 1
    # Should be much higher than random (motif influence factor is 0.15)
    assert count_rsi > 10, f"Motif-influenced RSI_14 trees: {count_rsi} out of {trials}" 