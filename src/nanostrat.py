"""
Nano-Strategy Tester for Project Crypto Bot Peter.

This module provides a lightweight, fast simulation environment (`run_nanostrat_test`)
for quickly evaluating the basic trigger logic of a single LogicDNA instance
against a small segment of market data. It's designed for rapid feedback
during the initial stages of DNA generation or mutation, focusing on trigger
activation and rudimentary P&L assessment without full portfolio simulation.
"""
import logging
from typing import List, Dict, Any, Union, TypedDict # Added TypedDict and ensured others

# Attempt to import LogicDNA, fallback to Any
try:
    from src.logic_dna import LogicDNA
except ImportError:
    LogicDNA = Any # type: ignore

logger = logging.getLogger(__name__) # Moved to module level

class NanoStratResult(TypedDict):
    activation_count: int
    virtual_pnl: float
    total_ticks: int

def run_nanostrat_test(dna_instance: LogicDNA, market_data_segment: List[Dict[str, Any]]) -> NanoStratResult:
    """
    Simulate the DNA's trigger condition against the market data segment.
    Args:
        dna_instance: LogicDNA instance
        market_data_segment: list of dicts, each with indicator values and price
    Returns:
        NanoStratResult: A TypedDict with activation_count, virtual_pnl, and total_ticks.
    """
    activation_count: int = 0
    virtual_pnl: float = 0.0
    total_ticks: int = len(market_data_segment)
    for i in range(total_ticks - 1):  # leave one for lookahead
        tick: Dict[str, Any] = market_data_segment[i]
        next_tick: Dict[str, Any] = market_data_segment[i+1]
        
        indicator_value: Any = tick.get(dna_instance.trigger_indicator, None)
        if indicator_value is None:
            continue
            
        op: str = dna_instance.trigger_operator
        threshold: Any = dna_instance.trigger_threshold # Type depends on DNA structure
        
        triggered: bool = False # Initialize before potential assignment
        if op == '<' and indicator_value < threshold:
            triggered = True
        elif op == '>' and indicator_value > threshold:
            triggered = True
            
        if triggered:
            activation_count += 1
            price_now: float = tick.get('price', 0.0)
            price_next: float = next_tick.get('price', 0.0)
            pnl: float = 0.0 # Initialize pnl
            if dna_instance.action_target.startswith('buy'):
                pnl = 1.0 if price_next > price_now else -1.0
            elif dna_instance.action_target.startswith('sell'):
                pnl = 1.0 if price_next < price_now else -1.0
            
            virtual_pnl += pnl
            logger.info(f"Nanostrat: DNA triggered at tick {i} | Action: {dna_instance.action_target} | PnL: {pnl}")
            
    logger.info(f"Nanostrat: DNA test complete | Activations: {activation_count} | VirtualPnL: {virtual_pnl} | Ticks: {total_ticks}")
    return {'activation_count': activation_count, 'virtual_pnl': virtual_pnl, 'total_ticks': total_ticks}