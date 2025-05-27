import pytest
from src.market_persona import MarketPersona

def test_market_persona_initialization():
    weights = {'metric_profit': 0.7, 'metric_trade_frequency': 0.2, 'complexity_penalty_weight': 0.1}
    persona = MarketPersona('Agile_HUNTER', weights)
    assert persona.name == 'Agile_HUNTER'
    assert persona.fitness_weights == weights

def test_calculate_fitness_hunter():
    weights = {'metric_profit': 0.7, 'metric_trade_frequency': 0.2, 'complexity_penalty_weight': 0.1}
    persona = MarketPersona('Agile_HUNTER', weights)
    metrics = {'metric_profit': 100, 'metric_trade_frequency': 10}
    complexity = 5
    fitness = persona.calculate_fitness(metrics, complexity)
    # (0.7*100) + (0.2*10) - (0.1*5) = 70 + 2 - 0.5 = 71.5
    assert pytest.approx(fitness, 0.01) == 71.5

def test_calculate_fitness_guardian():
    weights = {'metric_sharpe_ratio': 0.6, 'metric_max_drawdown': -0.3, 'complexity_penalty_weight': 0.2}
    persona = MarketPersona('Steady_GUARDIAN', weights)
    metrics = {'metric_sharpe_ratio': 2.0, 'metric_max_drawdown': 5.0}
    complexity = 4
    fitness = persona.calculate_fitness(metrics, complexity)
    # (0.6*2.0) + (-0.3*5.0) - (0.2*4) = 1.2 - 1.5 - 0.8 = -1.1
    assert pytest.approx(fitness, 0.01) == -1.1

def test_calculate_fitness_missing_metric():
    weights = {'metric_profit': 1.0, 'complexity_penalty_weight': 0.5}
    persona = MarketPersona('Profit_Only', weights)
    metrics = {}  # No profit metric present
    complexity = 3
    fitness = persona.calculate_fitness(metrics, complexity)
    # (1.0*0) - (0.5*3) = -1.5
    assert pytest.approx(fitness, 0.01) == -1.5

def test_calculate_fitness_zero_weights():
    weights = {'metric_profit': 0.0, 'complexity_penalty_weight': 0.0}
    persona = MarketPersona('Zero_Weights', weights)
    metrics = {'metric_profit': 100}
    complexity = 10
    fitness = persona.calculate_fitness(metrics, complexity)
    assert fitness == 0.0 