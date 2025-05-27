import json
import os
from datetime import datetime
from settings import GENERAL_SETTINGS

class DataLogger:
    """
    Utility for logging structured events as JSON Lines to a single .jsonl file.
    """
    def __init__(self, log_file=None):
        self.log_file = log_file or GENERAL_SETTINGS['STRUCTURED_DATA_LOG_FILE']
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def log_event(self, event_type: str, data_dict: dict):
        """
        Write a single event as a JSON object to the .jsonl file.
        Adds timestamp and event_type fields.
        Ensures JSON serializability and robust error handling.
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
        }
        event.update(self._make_json_serializable(data_dict))
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"DataLogger ERROR: Failed to write event to {self.log_file}: {e}")

    def _make_json_serializable(self, data):
        """
        Recursively convert data to JSON-serializable types.
        """
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(v) for v in data]
        elif hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
            return data.to_dict()
        elif hasattr(data, '__dict__'):
            return {k: self._make_json_serializable(v) for k, v in data.__dict__.items() if not k.startswith('_')}
        elif isinstance(data, (str, int, float, bool)) or data is None:
            return data
        else:
            return str(data)  # Fallback: convert to string

# Singleton instance for convenience
logger_instance = DataLogger()

def log_event(event_type: str, data_dict: dict):
    logger_instance.log_event(event_type, data_dict) 