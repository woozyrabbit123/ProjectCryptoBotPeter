import sys
import logging
from importlib.metadata import version, PackageNotFoundError
from packaging.version import parse as parse_version

def check_environment() -> None:
    """
    Check Python and key library versions. Raise RuntimeError if requirements are not met.
    """
    min_versions = {
        'python': (3, 10),
        'polars': '1.30.0',
        'numpy': '2.2.6',
        'aiohttp': '3.11.18',
        'scikit-learn': '1.6.1',
        'psutil': '7.0.0',
        'onnx': '1.18.0',
        'onnxruntime': '1.22.0',
        'orjson': '3.10.18',
        'certifi': '2025.4.26',
        'aiodns': '3.4.0',
        'packaging': '25.0',
    }
    # Python version
    if sys.version_info < min_versions['python']:
        raise RuntimeError(f"Python {min_versions['python']}+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    logging.info(f"Python version OK: {sys.version_info.major}.{sys.version_info.minor}")
    # Packages
    for pkg, min_ver in min_versions.items():
        if pkg == 'python':
            continue
        try:
            v = version(pkg)
            if parse_version(v) < parse_version(min_ver):
                raise RuntimeError(f"{pkg} >= {min_ver} required, found {v}")
            logging.info(f"{pkg} version OK: {v}")
        except PackageNotFoundError:
            raise RuntimeError(f"{pkg} not installed!") 