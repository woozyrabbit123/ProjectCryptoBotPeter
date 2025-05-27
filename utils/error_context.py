import contextvars
from typing import Any, Dict

_current_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar('_current_context', default={})

class ErrorContext:
    def __init__(self, context: Dict[str, Any]):
        self.token = None
        self.context = context
    def __enter__(self):
        self.token = _current_context.set(self.context)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            _current_context.reset(self.token)

def get_current_context() -> Dict[str, Any]:
    return _current_context.get({}) 