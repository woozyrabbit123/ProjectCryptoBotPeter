# Project Crypto Bot Peter - Performance Metrics API

## Purpose

This API provides access to performance metrics generated during the operational runs of Project Crypto Bot Peter. It allows users to retrieve summarized statistics from the performance log files via HTTP requests. Some endpoints may read from CSV log files, while others may use an SQLite database that also stores performance data.

## How to Run the API Server

To start the API server, navigate to the root directory of the project and execute the following command:

```bash
python api/main.py
```

The server will start, and by default, it listens on `0.0.0.0:5000`.

### Configuration

The host and port for the API server can be configured in the `config.ini` file located in the project's root directory. Modify or add the `[APISettings]` section:

```ini
[APISettings]
host = 0.0.0.0  ; Desired host IP address
port = 5000     ; Desired port number
```

If this section or these keys are not present in `config.ini`, the server will use the default values (`0.0.0.0` for host and `5000` for port).

The database path used by some endpoints is configured under `[DatabaseSettings]` in `config.ini`.

## Available Endpoints

### `GET /api/performance_metrics`

**Purpose:** Returns a JSON object containing calculated performance statistics derived from a specified performance log file (CSV).

**Query Parameters:**

*   `log_file_path` (optional): The path to the performance log CSV file.
    *   If provided, the API will attempt to read and process this file.
    *   **Example:** `/api/performance_metrics?log_file_path=logs/my_specific_run.csv`
    *   If not provided, the API defaults to using `performance_log_FULL_V1_2.csv`, which it expects to find in the project root directory (e.g., `/app/performance_log_FULL_V1_2.csv` in the sandbox environment).

**Example JSON Response:**

The structure of the JSON response will be similar to the following:

```json
{
    "log_file": "performance_log_FULL_V1_2.csv",
    "highest_fitness": 0.8512,
    "average_sharpe_ratio": 1.2345,
    "average_max_drawdown": 15.50,
    "average_winning_trade_percentage": 60.75,
    "average_trade_pl_overall": 0.0521,
    "total_unique_dna_structures": 150,
    "avg_fitness_per_generation": {
        "0": 0.5011,
        "1": 0.6123,
        "2": 0.6578
    },
    "dna_with_trades_count": 180,
    "evaluated_dna_count": 200
}
```
*(Note: `null` may be returned for metrics that could not be calculated, e.g., if the log file is empty or lacks certain data.)*

### `GET /api/top_n_performers`

**Purpose:** Retrieves the top N performing LogicDNA entries based on fitness score from the historical performance logs stored in the SQLite database.

**Query Parameters:**

*   `n` (optional):
    *   **Name:** `n`
    *   **Description:** The number of top performers to retrieve.
    *   **Default:** `10` if not provided, or if an invalid (non-integer, non-positive) value is given.
    *   **Example:** `/api/top_n_performers?n=5`

**Example JSON Response:**

A JSON array of objects, where each object represents a top-performing DNA entry.

```json
[
    {
        "dna_id": "dna_xyz",
        "fitness_score": 0.95,
        "logic_dna_structure_representation": "Structure_Details_Here",
        "active_persona_name": "AggressiveGrowth",
        "timestamp_of_evaluation": "YYYY-MM-DDTHH:MM:SS.ffffff"
    },
    {
        "dna_id": "dna_abc",
        "fitness_score": 0.92,
        "logic_dna_structure_representation": "Another_Structure",
        "active_persona_name": "Conservative",
        "timestamp_of_evaluation": "YYYY-MM-DDTHH:MM:SS.ffffff"
    }
    // ... more entries up to n
]
```
*(Note: The list will be empty if no performance data is found in the database. The columns returned are: `dna_id`, `fitness_score`, `logic_dna_structure_representation`, `active_persona_name`, `timestamp_of_evaluation`.)*
