import logging

def run_nanostrat_test(dna_instance, market_data_segment):
    """
    Simulate the DNA's trigger condition against the market data segment.
    Args:
        dna_instance: LogicDNA instance
        market_data_segment: list of dicts, each with indicator values and price
    Returns:
        dict: {'activation_count': int, 'virtual_pnl': float, 'total_ticks': int}
    """
    logger = logging.getLogger(__name__)
    activation_count = 0
    virtual_pnl = 0.0
    total_ticks = len(market_data_segment)
    for i in range(total_ticks - 1):  # leave one for lookahead
        tick = market_data_segment[i]
        next_tick = market_data_segment[i+1]
        indicator_value = tick.get(dna_instance.trigger_indicator, None)
        if indicator_value is None:
            continue
        op = dna_instance.trigger_operator
        threshold = dna_instance.trigger_threshold
        triggered = (op == '<' and indicator_value < threshold) or (op == '>' and indicator_value > threshold)
        if triggered:
            activation_count += 1
            # Simple PnL: if action is buy and price goes up next tick, +1; else -1
            price_now = tick.get('price', 0.0)
            price_next = next_tick.get('price', 0.0)
            if dna_instance.action_target.startswith('buy'):
                pnl = 1.0 if price_next > price_now else -1.0
            elif dna_instance.action_target.startswith('sell'):
                pnl = 1.0 if price_next < price_now else -1.0
            else:
                pnl = 0.0
            virtual_pnl += pnl
            logger.info(f"Nanostrat: DNA triggered at tick {i} | Action: {dna_instance.action_target} | PnL: {pnl}")
    logger.info(f"Nanostrat: DNA test complete | Activations: {activation_count} | VirtualPnL: {virtual_pnl} | Ticks: {total_ticks}")
    return {'activation_count': activation_count, 'virtual_pnl': virtual_pnl, 'total_ticks': total_ticks} 