"""
Market Persona for Project Crypto Bot Peter.

This module defines the `MarketPersona` class, which represents different
market condition biases or trading styles (e.g., 'Agile_HUNTER',
'Steady_GUARDIAN'). Each persona has a specific fitness function, defined
by a set of weights for various performance metrics and complexity penalties.
This allows the evolutionary system to optimize LogicDNA instances for
different market interpretations or objectives.
"""
from typing import Dict

class MarketPersona:
    """
    Represents a market persona with a specific fitness function for evaluating LogicDNA individuals.
    Attributes:
        name (str): The persona's name (e.g., 'Agile_HUNTER', 'Steady_GUARDIAN').
        fitness_weights (Dict[str, float]): Weights for each metric and the complexity penalty (e.g., {'metric_profit': 0.6, 'metric_trade_frequency': 0.3, 'complexity_penalty_weight': 0.1}).
    """
    name: str
    fitness_weights: Dict[str, float]

    def __init__(self, name: str, fitness_weights: Dict[str, float]): # fitness_weights typed
        self.name: str = name # Ensure self.name is typed
        self.fitness_weights: Dict[str, float] = fitness_weights.copy() if fitness_weights else {}

    def calculate_fitness(self, performance_metrics: Dict[str, float], complexity_score: float) -> float: # performance_metrics typed
        """
        Calculate the fitness value for a LogicDNA individual.
        Args:
            performance_metrics (Dict[str, float]): Dictionary of performance metrics (e.g., {'metric_profit': X, ...}).
            complexity_score (float): The complexity score (e.g., weighted sum of node_count and tree_depth).
        Returns:
            float: The calculated fitness value.
        """
        fitness: float = 0.0
        for metric, weight_untyped in self.fitness_weights.items():
            metric: str = metric # Explicitly typing loop variable
            weight: float = float(weight_untyped) # Explicitly typing loop variable and casting
            if metric == 'complexity_penalty_weight':
                continue  # Handle complexity penalty separately
            value: float = performance_metrics.get(metric, 0.0) # value from dict.get can be float
            fitness += weight * value
        # Subtract weighted complexity penalty if present
        penalty_weight: float = self.fitness_weights.get('complexity_penalty_weight', 0.0) # penalty_weight is float
        fitness -= penalty_weight * complexity_score
        return fitness 