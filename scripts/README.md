# Performance Report Script

This directory contains scripts for analyzing and reporting on the performance of the Logic Evolution Engine (LEE).

## `generate_simple_performance_report.py`

### Purpose

The `generate_simple_performance_report.py` script is designed to process performance log files (typically CSV format) generated by the LEE system. It provides a comprehensive summary of an evolutionary run.

The script performs the following actions:
1.  Reads a specified performance log CSV file.
2.  Calculates and aggregates key summary statistics. The report includes:
    *   The highest fitness score achieved overall.
    *   The average fitness score for each generation.
    *   The total number of unique DNA structures observed.
    *   **Advanced Performance Metrics (averaged across applicable DNA individuals):**
        *   Overall Sharpe Ratio.
        *   Maximum Drawdown (as a percentage).
        *   Winning Trade Percentage.
        *   Average Trade Profit/Loss (per unit, based on BUY/SELL pairs).
    *   **Motif Usage Analysis:**
        *   Identifies the top 3 most frequently occurring indicator-based motifs. The regex pattern used for discovering these motifs (e.g., to find patterns like 'Indicator_EMA_20_Used') can be customized in `config.ini` under the `[MotifAnalysis]` section using the `motif_regex_pattern` key. It defaults to a predefined pattern (`Indicator_([A-Z0-9]+)_(\d+)(?:_Used)?`) if this configuration is not specified.
        *   Reports their frequency.
        *   Compares the average fitness of DNAs containing these motifs against the overall average fitness.
        *   Includes a "Motif-Regime Correlation Hints" section, which provides insights into the typical market conditions (CES vectors from the `ces_vector_at_evaluation_time` column in the log) associated with the top DNA motifs by showing the most frequent CES vector for each top motif.
3.  Generates a consolidated report of these statistics.
4.  **HTML Dashboard Generation**: Generates a simple HTML dashboard (`reports/latest_performance_dashboard.html`) which includes summary statistics and embeds data for a potential equity curve visualization.

### Output Format

The script outputs its primary findings directly to the console in **Markdown format**. This allows for easy readability and can be copied into Markdown-compatible documents or wikis. Additionally, it saves an HTML dashboard as mentioned above.

### Data Sources

Currently, this script primarily reads data from CSV log files. Note that the main LEE application (`src/lee.py`) now logs performance data to both CSV files and an SQLite database (`data/performance_logs.db` by default, configurable in `config.ini`). While this script does not yet directly query the SQLite database, future enhancements could leverage it for more complex queries or broader historical analysis.

### How to Run

To execute the script, navigate to the root directory of the project and use the following command structure:

**Basic Command (using default log file):**

```bash
python scripts/generate_simple_performance_report.py
```

This command will attempt to read the default performance log file, which is `performance_log_FULL_V1_2.csv`, expected to be located in the project root directory (`/app/performance_log_FULL_V1_2.csv` in the sandbox environment). It will also generate the HTML report in the `reports/` directory.

**Specifying a Log File Path:**

You can specify a different log file using the `--log_file_path` argument:

```bash
python scripts/generate_simple_performance_report.py --log_file_path /path/to/your/performance_log.csv
```

Replace `/path/to/your/performance_log.csv` with the actual path to your CSV file.

**Example:**

If your log file is named `my_experiment_log.csv` and is located in a directory called `lee_results` within your home directory, the command might look like:

```bash
python scripts/generate_simple_performance_report.py --log_file_path ~/lee_results/my_experiment_log.csv
```

### Configurability

The script's internal logging behavior (e.g., verbosity of its own processing messages, not the content of the generated report) can be configured via a `config.ini` file placed in the project's root directory. Specifically, the `level` setting under the `[Logging]` section in `config.ini` will be used if the `src.utils.logging_utils` module is available to the script. If not found, basic logging is used.

Additionally, as mentioned in the "Motif Usage Analysis" section, the regex pattern for motif identification is configurable via `config.ini`. The database path for the main application is also configured in `config.ini` (`[DatabaseSettings] -> db_path`).
