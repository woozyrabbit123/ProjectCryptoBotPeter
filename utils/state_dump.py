import pickle
import time
import logging
from typing import Any

def dump_bot_state(fsm_instance: Any, micro_regime_detector_instance: Any, shard_learner_instance: Any, filename_prefix: str = "state_snapshot_") -> str:
    """
    Pickle and save a snapshot of key bot state attributes to a timestamped file.
    Failure Modes: May skip unpicklable objects or large data. Logs errors.
    Performance Notes: Intended for on-demand debugging, not high-frequency use.
    Returns the path to the dump file.
    """
    state = {}
    try:
        state['fsm'] = {
            'current_state': getattr(fsm_instance, 'state', None),
            'buffer_sample': getattr(fsm_instance, 'buffer', None)[:10] if hasattr(fsm_instance, 'buffer') else None,
            'current_ema': getattr(fsm_instance, 'current_ema', None),
        }
    except Exception as e:
        logging.error(f"FSM state dump error: {e}")
    try:
        state['micro_regime_detector'] = {
            'last_ema': getattr(micro_regime_detector_instance, 'last_ema', None),
            'last_regime_signal': getattr(micro_regime_detector_instance, 'last_regime_signal', None),
        }
    except Exception as e:
        logging.error(f"MicroRegimeDetector state dump error: {e}")
    try:
        state['shard_learner'] = {
            'lof_model': getattr(shard_learner_instance, 'lof_model', None),
            'norm_stats': getattr(shard_learner_instance, 'norm_stats', None),
            'recent_anomaly_reports': getattr(shard_learner_instance, 'recent_anomaly_reports', None),
        }
    except Exception as e:
        logging.error(f"ShardLearner state dump error: {e}")
    ts = int(time.time())
    fname = f"{filename_prefix}{ts}.pkl"
    try:
        with open(fname, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"State snapshot saved to {fname}")
        return fname
    except Exception as e:
        logging.error(f"Failed to save state snapshot: {e}")
        return "" 