"""
Market Persona for Project Crypto Bot Peter.

This module defines the `MarketPersona` class, which represents different
market condition biases or trading styles (e.g., 'Agile_HUNTER',
'Steady_GUARDIAN'). Each persona has a specific fitness function, defined
by a set of weights for various performance metrics and complexity penalties.
This allows the evolutionary system to optimize LogicDNA instances for
different market interpretations or objectives.
"""
class MarketPersona:
    """
    Represents a market persona with a specific fitness function for evaluating LogicDNA individuals.
    Attributes:
        name (str): The persona's name (e.g., 'Agile_HUNTER', 'Steady_GUARDIAN').
        fitness_weights (dict): Weights for each metric and the complexity penalty (e.g., {'metric_profit': 0.6, 'metric_trade_frequency': 0.3, 'complexity_penalty_weight': 0.1}).
    """
    def __init__(self, name: str, fitness_weights: dict):
        self.name = name
        self.fitness_weights = fitness_weights.copy() if fitness_weights else {}

    def calculate_fitness(self, performance_metrics: dict, complexity_score: float) -> float:
        """
        Calculate the fitness value for a LogicDNA individual.
        Args:
            performance_metrics (dict): Dictionary of performance metrics (e.g., {'metric_profit': X, ...}).
            complexity_score (float): The complexity score (e.g., weighted sum of node_count and tree_depth).
        Returns:
            float: The calculated fitness value.
        """
        fitness = 0.0
        for metric, weight in self.fitness_weights.items():
            if metric == 'complexity_penalty_weight':
                continue  # Handle complexity penalty separately
            value = performance_metrics.get(metric, 0.0)
            fitness += weight * value
        # Subtract weighted complexity penalty if present
        penalty_weight = self.fitness_weights.get('complexity_penalty_weight', 0.0)
        fitness -= penalty_weight * complexity_score
        return fitness 