# Project Crypto Bot Peter

A resource-efficient cryptocurrency trading bot MVP designed to run on constrained hardware (AMD Ryzen 5 5600H, NVIDIA RTX 3050 4GB VRAM, ~5.86GB system RAM).

## Project Structure

```
project_crypto_bot_peter/
├── data/                     # For Parquet files, shard logs, etc.
├── models/                   # For ONNX, TensorRT plans
├── src/
│   ├── data_handling.py     # Market data loading and processing
│   ├── feature_engineering.py # Technical indicators and features
│   ├── model_inference.py   # PyTorch model and TensorRT inference
│   ├── trading_logic.py     # Trading strategies
│   ├── fsm_sentinel.py      # Resource management FSM
│   ├── shard_logger.py      # Efficient binary logging
│   └── bot_scheduler.py     # Execution scheduling
├── tests/                    # Unit tests
├── config/                   # Configuration files
├── main.py                  # Main entry point
└── requirements.txt         # Python dependencies
```

## Setup Instructions

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix/MacOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install TensorRT:
   - Download and install TensorRT from NVIDIA's website
   - Follow NVIDIA's installation guide for your specific system
   - Note: Direct pip installation might not be possible; manual installation may be required

## Usage

Run the bot in monitoring mode (L0):
```bash
python main.py --mode L0
```

Run the bot in trading mode (L1):
```bash
python main.py --mode L1
```

## Development

- Use Python 3.9 or higher
- Follow PEP 8 style guide
- Run tests with `pytest`
- Format code with `black`
- Type check with `mypy`

## Notes

- The bot is designed for resource efficiency
- Uses Polars for efficient data handling
- Implements a "Sleepy Sentinel" FSM for resource management
- TensorRT for optimized model inference
- Binary shard logging for efficient data storage 