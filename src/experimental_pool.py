import logging
import math
from typing import List, Dict, Tuple, Any, Optional # Added Optional for consistency
# Attempt to import LogicDNA
try:
    from src.logic_dna import LogicDNA
except ImportError:
    LogicDNA = Any # type: ignore

from utils.hardware_monitor import get_cpu_load
from settings import EPM_SETTINGS
from src.data_logger import log_event


def check_settings_dict(settings_dict: Dict[str, Any], required_keys: List[str], dict_name: str) -> None:
    missing = [k for k in required_keys if k not in settings_dict]
    if missing:
        log_event('CRITICAL_ERROR', {'missing_keys': missing, 'settings_dict': dict_name})
        # Consider logging the error as well, or ensure log_event does that.
        raise RuntimeError(f"CRITICAL: Missing keys in {dict_name}: {missing}")

class PoolDNAEntry:
    """
    Represents a DNA entry in the experimental pool, tracking its performance and activation history.
    """
    dna: LogicDNA
    ticks_in_pool: int
    rolling_pnl_pool: float
    activation_count_pool: int
    pnl_history: List[float]

    def __init__(self, dna: LogicDNA) -> None:
        self.dna: LogicDNA = dna
        self.ticks_in_pool: int = 0
        self.rolling_pnl_pool: float = 0.0
        self.activation_count_pool: int = 0
        self.pnl_history: List[float] = []  # For stddev calculation

class ExperimentalPoolManager:
    """
    Manages the pool of experimental DNAs, including evaluation, pruning, and graduation.
    """
    pool: List[PoolDNAEntry]
    logger: logging.Logger # Explicit type hint for logger

    def __init__(self) -> None:
        # Defensive check for EPM_SETTINGS
        required_keys: List[str] = ['MAX_POOL_SIZE', 'MAX_POOL_LIFE', 'MIN_PERFORMANCE_THRESHOLD', 'MIN_POOL_SIZE', 'GRADUATION_MIN_SHARPE_POOL', 'GRADUATION_MIN_PNL_POOL', 'GRADUATION_MIN_TICKS_IN_POOL', 'GRADUATION_MIN_ACTIVATIONS_POOL', 'GRADUATION_EPSILON']
        check_settings_dict(EPM_SETTINGS, required_keys, 'EPM_SETTINGS')
        self.pool: List[PoolDNAEntry] = [] 
        self.logger: logging.Logger = logging.getLogger(__name__)

    def add_dna_to_pool(self, dna: LogicDNA) -> None:
        """
        Add a new DNA to the pool, pruning the worst if the pool is full.
        """
        if self.get_pool_size() >= EPM_SETTINGS['MAX_POOL_SIZE']:
            if not self.pool: # Should not happen if size >= MAX_POOL_SIZE > 0, but good for safety
                self.logger.warning("EPM: Pool is full but empty, cannot discard.") # Should ideally not happen
            else:
                worst_idx: int = min(range(len(self.pool)), key=lambda i: self.pool[i].rolling_pnl_pool)
                self.logger.info(f"EPM: Pool full. Discarding worst DNA: {self.pool[worst_idx].dna.id if hasattr(self.pool[worst_idx].dna, 'id') else self.pool[worst_idx].dna}")
                self.pool.pop(worst_idx)
        
        self.pool.append(PoolDNAEntry(dna))
        self.logger.info(f"EPM: Added DNA to pool: {dna.id if hasattr(dna, 'id') else dna}")


    def evaluate_pool_tick(self, market_tick: Dict[str, Any]) -> None:
        """
        Evaluate all DNAs in the pool for the current market tick, updating PnL and activations.
        """
        for entry in self.pool: # entry is PoolDNAEntry
            entry.ticks_in_pool += 1
            indicator_value: Optional[Any] = market_tick.get(entry.dna.trigger_indicator, None)
            if indicator_value is None:
                continue
            op: str = entry.dna.trigger_operator
            threshold: Any = entry.dna.trigger_threshold # Type depends on DNA structure
            triggered: bool = (op == '<' and indicator_value < threshold) or \
                              (op == '>' and indicator_value > threshold)
            if triggered:
                entry.activation_count_pool += 1
                price_now: float = market_tick.get('price', 0.0)
                price_prev: float = market_tick.get('price_prev', price_now)
                pnl: float = 0.0
                if entry.dna.action_target.startswith('buy'):
                    pnl = 1.0 if price_now > price_prev else -1.0
                elif entry.dna.action_target.startswith('sell'):
                    pnl = 1.0 if price_now < price_prev else -1.0
                
                entry.rolling_pnl_pool += pnl
                entry.pnl_history.append(pnl)
                self.logger.info(f"EPM: DNA {entry.dna.id if hasattr(entry.dna, 'id') else entry.dna} triggered | PnL: {pnl} | RollingPnL: {entry.rolling_pnl_pool}")

    def prune_pool(self) -> None:
        """
        Prune the pool based on hardware constraints and performance/inactivity.
        Adds error handling for get_cpu_load and logs warnings on failure.
        """
        cpu_load: float
        try:
            cpu_load = get_cpu_load()
        except Exception as e:
            self.logger.warning(f"EPM: Failed to get CPU load: {e}. Using safe default (99.0)")
            cpu_load = 99.0
        
        if cpu_load > 85.0 and self.pool:
            worst_idx: int = min(range(len(self.pool)), key=lambda i: self.pool[i].rolling_pnl_pool)
            pruned_entry: PoolDNAEntry = self.pool[worst_idx]
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
            self.logger.info(f"EPM: Pruning for hardware. Removing DNA: {pruned_entry.dna.id if hasattr(pruned_entry.dna, 'id') else pruned_entry.dna}")
            self.pool.pop(worst_idx)
        
        to_remove: List[int] = []
        for i, entry in enumerate(self.pool): # entry is PoolDNAEntry
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
                    self.logger.info(f"EPM: Pruning for poor performance. Removing DNA: {entry.dna.id if hasattr(entry.dna, 'id') else entry.dna}")
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
                    self.logger.info(f"EPM: Pruning for inactivity. Removing DNA: {entry.dna.id if hasattr(entry.dna, 'id') else entry.dna}")
                    to_remove.append(i)
            except Exception as e:
                self.logger.warning(f"EPM: Error during pool pruning for DNA {entry.dna.id if hasattr(entry.dna, 'id') else entry.dna}: {e}")
        for idx in reversed(to_remove):
            self.pool.pop(idx)

    def get_pool_size(self) -> int:
        """
        Returns the current size of the pool.
        """
        return len(self.pool)

    def get_average_pool_pnl(self) -> float:
        """
        Returns the average rolling PnL of the pool.
        """
        if not self.pool:
            return 0.0
        return sum(entry.rolling_pnl_pool for entry in self.pool) / len(self.pool)

    def check_for_graduates(self) -> List[Tuple[int, PoolDNAEntry]]:
        """
        Check for DNAs eligible for graduation based on performance and activity.
        Returns:
            list: List of (index, PoolDNAEntry) tuples for graduates.
        """
        graduates: List[Tuple[int, PoolDNAEntry]] = []
        for idx, entry in enumerate(self.pool): # entry is PoolDNAEntry
            if entry.ticks_in_pool < EPM_SETTINGS['GRADUATION_MIN_TICKS_IN_POOL']:
                continue
            if entry.activation_count_pool < EPM_SETTINGS['GRADUATION_MIN_ACTIVATIONS_POOL']:
                continue
            if entry.rolling_pnl_pool < EPM_SETTINGS['GRADUATION_MIN_PNL_POOL']:
                continue
            
            mean_pnl: float = sum(entry.pnl_history) / (len(entry.pnl_history) or 1)
            variance: float = sum((x - mean_pnl) ** 2 for x in entry.pnl_history) / (len(entry.pnl_history) or 1)
            stddev: float = math.sqrt(variance) + EPM_SETTINGS['GRADUATION_EPSILON']
            sharpe: float = mean_pnl / stddev
            if sharpe >= EPM_SETTINGS['GRADUATION_MIN_SHARPE_POOL']:
                self.logger.info(f"EPM: DNA eligible for graduation: {entry.dna.id if hasattr(entry.dna, 'id') else entry.dna} | Sharpe={sharpe:.2f} | PnL={entry.rolling_pnl_pool:.2f}")
                graduates.append((idx, entry))
        return graduates

    def pop_graduate(self, idx: int) -> PoolDNAEntry:
        """
        Remove and return a graduated DNA from the pool.
        Args:
            idx (int): Index of the graduate in the pool.
        Returns:
            PoolDNAEntry: The graduated entry.
        """
        entry: PoolDNAEntry = self.pool.pop(idx)
        self.logger.info(f"EPM: DNA graduated and removed from pool: {entry.dna.id if hasattr(entry.dna, 'id') else entry.dna}")
        return entry 