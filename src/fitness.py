import logging

def evaluate_dna_fitness(nanostrat_results, cpu_load):
    logger = logging.getLogger(__name__)
    if cpu_load > 85.0:
        logger.info(f"Fitness: Discarded due to high CPU load ({cpu_load:.2f}%)")
        return 'discard_hardware_limit'
    if nanostrat_results['virtual_pnl'] > 0 and nanostrat_results['activation_count'] >= 1:
        logger.info(f"Fitness: DNA survived MVL (PnL={nanostrat_results['virtual_pnl']}, Activations={nanostrat_results['activation_count']})")
        return 'survived_mvl'
    logger.info(f"Fitness: DNA discarded for poor performance (PnL={nanostrat_results['virtual_pnl']}, Activations={nanostrat_results['activation_count']})")
    return 'discard_performance' 