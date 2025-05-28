"""
Fitness Evaluation for Project Crypto Bot Peter.

This module provides functions to evaluate the fitness of a trading strategy
(represented as LogicDNA) based on its performance in simulations (e.g.,
nanostrat_results) and other factors like system resource usage (e.g., CPU load).
The fitness score helps determine which strategies are promising and should be
further explored or promoted by the evolutionary algorithms.
"""
import logging
from typing import Dict, Any

# Attempt to import NanoStratResult, fallback to Dict[str, Any]
try:
    from src.nanostrat import NanoStratResult
except ImportError:
    NanoStratResult = Dict[str, Any] # type: ignore

logger = logging.getLogger(__name__) # Moved to module level

def evaluate_dna_fitness(nanostrat_results: NanoStratResult, cpu_load: float) -> str:
    if cpu_load > 85.0:
        logger.info(f"Fitness: Discarded due to high CPU load ({cpu_load:.2f}%)")
        return 'discard_hardware_limit'
    if nanostrat_results['virtual_pnl'] > 0 and nanostrat_results['activation_count'] >= 1:
        logger.info(f"Fitness: DNA survived MVL (PnL={nanostrat_results['virtual_pnl']}, Activations={nanostrat_results['activation_count']})")
        return 'survived_mvl'
    logger.info(f"Fitness: DNA discarded for poor performance (PnL={nanostrat_results['virtual_pnl']}, Activations={nanostrat_results['activation_count']})")
    return 'discard_performance' 