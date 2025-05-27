import logging
import math
from src.logic_dna import LogicDNA
from utils.hardware_monitor import get_cpu_load
from settings import EPM_SETTINGS
from src.data_logger import log_event

def check_settings_dict(settings_dict, required_keys, dict_name):
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        raise RuntimeError(f"CRITICAL: Missing keys in {dict_name}: {missing}")

class PoolDNAEntry:
    """
    Represents a DNA entry in the experimental pool, tracking its performance and activation history.
    """
    def __init__(self, dna):
        self.dna = dna
        self.ticks_in_pool = 0
        self.rolling_pnl_pool = 0.0
        self.activation_count_pool = 0
        self.pnl_history = []  # For stddev calculation

class ExperimentalPoolManager:
    """
    Manages the pool of experimental DNAs, including evaluation, pruning, and graduation.
    """
    def __init__(self):
        # Defensive check for EPM_SETTINGS
        required_keys = ['MAX_POOL_SIZE', 'MAX_POOL_LIFE', 'MIN_PERFORMANCE_THRESHOLD', 'MIN_POOL_SIZE', 'GRADUATION_MIN_SHARPE_POOL', 'GRADUATION_MIN_PNL_POOL', 'GRADUATION_MIN_TICKS_IN_POOL', 'GRADUATION_MIN_ACTIVATIONS_POOL', 'GRADUATION_EPSILON']
        check_settings_dict(EPM_SETTINGS, required_keys, 'EPM_SETTINGS')
        self.pool = []  # List of PoolDNAEntry
        self.logger = logging.getLogger(__name__)

    def add_dna_to_pool(self, dna):
        """
        Add a new DNA to the pool, pruning the worst if the pool is full.
        """
        if self.get_pool_size() < EPM_SETTINGS['MAX_POOL_SIZE']:
            self.pool.append(PoolDNAEntry(dna))
            self.logger.info(f"EPM: Added DNA to pool: {dna}")
        else:
            # Discard worst (lowest rolling_pnl_pool)
            worst_idx = min(range(len(self.pool)), key=lambda i: self.pool[i].rolling_pnl_pool)
            self.logger.info(f"EPM: Pool full. Discarding worst DNA: {self.pool[worst_idx].dna}")
            self.pool.pop(worst_idx)
            self.pool.append(PoolDNAEntry(dna))
            self.logger.info(f"EPM: Added DNA to pool: {dna}")

    def evaluate_pool_tick(self, market_tick):
        """
        Evaluate all DNAs in the pool for the current market tick, updating PnL and activations.
        """
        for entry in self.pool:
            entry.ticks_in_pool += 1
            indicator_value = market_tick.get(entry.dna.trigger_indicator, None)
            if indicator_value is None:
                continue
            op = entry.dna.trigger_operator
            threshold = entry.dna.trigger_threshold
            triggered = (op == '<' and indicator_value < threshold) or (op == '>' and indicator_value > threshold)
            if triggered:
                entry.activation_count_pool += 1
                price_now = market_tick.get('price', 0.0)
                # For pool, we don't have lookahead, so use price change from previous tick if available
                # For MVP, just use 0.0 (no PnL) if not available
                price_prev = market_tick.get('price_prev', price_now)
                if entry.dna.action_target.startswith('buy'):
                    pnl = 1.0 if price_now > price_prev else -1.0
                elif entry.dna.action_target.startswith('sell'):
                    pnl = 1.0 if price_now < price_prev else -1.0
                else:
                    pnl = 0.0
                entry.rolling_pnl_pool += pnl
                entry.pnl_history.append(pnl)
                self.logger.info(f"EPM: DNA {entry.dna} triggered | PnL: {pnl} | RollingPnL: {entry.rolling_pnl_pool}")

    def prune_pool(self):
        """
        Prune the pool based on hardware constraints and performance/inactivity.
        Adds error handling for get_cpu_load and logs warnings on failure.
        """
        try:
            cpu_load = get_cpu_load()
        except Exception as e:
            self.logger.warning(f"EPM: Failed to get CPU load: {e}. Using safe default (99.0)")
            cpu_load = 99.0
        # Hardware pruning
        if cpu_load > 85.0 and self.pool:
            worst_idx = min(range(len(self.pool)), key=lambda i: self.pool[i].rolling_pnl_pool)
            pruned_entry = self.pool[worst_idx]
            log_event(
                'EPM_DNA_PRUNED',
                {
                    'dna_id': pruned_entry.dna.id,
                    'parent_id': pruned_entry.dna.parent_id,
                    'seed_type': pruned_entry.dna.seed_type,
                    'reason': 'hardware',
                    'epm_performance_summary': {
                        'rolling_pnl_pool': pruned_entry.rolling_pnl_pool,
                        'activation_count_pool': pruned_entry.activation_count_pool,
                        'ticks_in_pool': pruned_entry.ticks_in_pool,
                    },
                    'system_cpu_load': cpu_load,
                    'pool_size_before': len(self.pool),
                }
            )
            self.logger.info(f"EPM: Pruning for hardware. Removing DNA: {pruned_entry.dna}")
            self.pool.pop(worst_idx)
        # Performance pruning
        to_remove = []
        for i, entry in enumerate(self.pool):
            try:
                if entry.ticks_in_pool > EPM_SETTINGS['MAX_POOL_LIFE'] and entry.rolling_pnl_pool < EPM_SETTINGS['MIN_PERFORMANCE_THRESHOLD']:
                    log_event(
                        'EPM_DNA_PRUNED',
                        {
                            'dna_id': entry.dna.id,
                            'parent_id': entry.dna.parent_id,
                            'seed_type': entry.dna.seed_type,
                            'reason': 'performance',
                            'epm_performance_summary': {
                                'rolling_pnl_pool': entry.rolling_pnl_pool,
                                'activation_count_pool': entry.activation_count_pool,
                                'ticks_in_pool': entry.ticks_in_pool,
                            },
                            'system_cpu_load': cpu_load,
                            'pool_size_before': len(self.pool),
                        }
                    )
                    self.logger.info(f"EPM: Pruning for poor performance. Removing DNA: {entry.dna}")
                    to_remove.append(i)
                elif entry.ticks_in_pool > EPM_SETTINGS['MAX_POOL_LIFE'] and entry.activation_count_pool < 2:
                    log_event(
                        'EPM_DNA_PRUNED',
                        {
                            'dna_id': entry.dna.id,
                            'parent_id': entry.dna.parent_id,
                            'seed_type': entry.dna.seed_type,
                            'reason': 'inactivity',
                            'epm_performance_summary': {
                                'rolling_pnl_pool': entry.rolling_pnl_pool,
                                'activation_count_pool': entry.activation_count_pool,
                                'ticks_in_pool': entry.ticks_in_pool,
                            },
                            'system_cpu_load': cpu_load,
                            'pool_size_before': len(self.pool),
                        }
                    )
                    self.logger.info(f"EPM: Pruning for inactivity. Removing DNA: {entry.dna}")
                    to_remove.append(i)
            except Exception as e:
                self.logger.warning(f"EPM: Error during pool pruning for DNA {entry.dna}: {e}")
        for idx in reversed(to_remove):
            self.pool.pop(idx)

    def get_pool_size(self):
        """
        Returns the current size of the pool.
        """
        return len(self.pool)

    def get_average_pool_pnl(self):
        """
        Returns the average rolling PnL of the pool.
        """
        if not self.pool:
            return 0.0
        return sum(entry.rolling_pnl_pool for entry in self.pool) / len(self.pool)

    def check_for_graduates(self):
        """
        Check for DNAs eligible for graduation based on performance and activity.
        Returns:
            list: List of (index, PoolDNAEntry) tuples for graduates.
        """
        graduates = []
        for idx, entry in enumerate(self.pool):
            if entry.ticks_in_pool < EPM_SETTINGS['GRADUATION_MIN_TICKS_IN_POOL']:
                continue
            if entry.activation_count_pool < EPM_SETTINGS['GRADUATION_MIN_ACTIVATIONS_POOL']:
                continue
            if entry.rolling_pnl_pool < EPM_SETTINGS['GRADUATION_MIN_PNL_POOL']:
                continue
            # Calculate Sharpe: mean / stddev
            mean_pnl = sum(entry.pnl_history) / (len(entry.pnl_history) or 1)
            variance = sum((x - mean_pnl) ** 2 for x in entry.pnl_history) / (len(entry.pnl_history) or 1)
            stddev = math.sqrt(variance) + EPM_SETTINGS['GRADUATION_EPSILON']
            sharpe = mean_pnl / stddev
            if sharpe >= EPM_SETTINGS['GRADUATION_MIN_SHARPE_POOL']:
                self.logger.info(f"EPM: DNA eligible for graduation: {entry.dna} | Sharpe={sharpe:.2f} | PnL={entry.rolling_pnl_pool:.2f}")
                graduates.append((idx, entry))
        return graduates

    def pop_graduate(self, idx):
        """
        Remove and return a graduated DNA from the pool.
        Args:
            idx (int): Index of the graduate in the pool.
        Returns:
            PoolDNAEntry: The graduated entry.
        """
        entry = self.pool.pop(idx)
        self.logger.info(f"EPM: DNA graduated and removed from pool: {entry.dna}")
        return entry 