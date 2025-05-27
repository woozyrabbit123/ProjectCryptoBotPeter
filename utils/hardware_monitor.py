import psutil
import logging

def get_cpu_load():
    """Return current system CPU load percentage (float)."""
    cpu = psutil.cpu_percent(interval=0.1)
    logging.getLogger(__name__).info(f"HardwareMonitor: CPU load = {cpu:.2f}%")
    return cpu 