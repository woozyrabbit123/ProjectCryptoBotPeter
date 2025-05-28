import argparse
import logging
import os
import pandas as pd
import json
import numpy as np
import re # Added for regex
from collections import Counter # Added for frequency counting
from typing import Optional, List, Dict, Any, Tuple
import configparser # Added for config.ini handling

# Attempt to import setup_global_logging, handle if not found for standalone execution
try:
    from src.utils.logging_utils import setup_global_logging
    # Call at the beginning of the script
    setup_global_logging('config.ini') # Assumes config.ini is in the root for the script
except ImportError:
    print("Warning: src.utils.logging_utils not found. Using basic logging.")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Import refactored utility functions
from src.utils.performance_metrics_utils import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_trade_stats
)

logger: logging.Logger = logging.getLogger(__name__)

# --- Motif Analysis Function ---
def analyze_motif_usage(df: pd.DataFrame) -> str:
    """
    Analyzes motif usage, identifies top 3 motifs, compares their average fitness,
    and analyzes correlation with CES vectors.
    Returns a Markdown formatted string of the analysis.
    """
    motif_lines: List[str] = ["\n## Motif Usage Analysis"]

    # --- Motif Regex Configuration ---
    default_motif_regex: str = r"Indicator_([A-Z0-9]+)_(\d+)(?:_Used)?"
    motif_regex_to_use: str = default_motif_regex

    config = configparser.ConfigParser()
    # Path to config.ini assumes script is in 'scripts/' and config.ini is in project root.
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')

    if os.path.exists(config_path):
        config.read(config_path)
        try:
            configured_regex = config.get('MotifAnalysis', 'motif_regex_pattern', fallback=default_motif_regex)
            if configured_regex and configured_regex.strip(): # Ensure not empty or just whitespace
                motif_regex_to_use = configured_regex
                if motif_regex_to_use == default_motif_regex:
                    logger.info(f"Motif Analysis: Using default regex pattern (also found in config.ini or fallback used): {motif_regex_to_use}")
                else:
                    logger.info(f"Motif Analysis: Using regex pattern from config.ini: {motif_regex_to_use}")
            else: # Empty value in config
                logger.warning(f"Motif Analysis: 'motif_regex_pattern' in config.ini [MotifAnalysis] is empty. Using default: {default_motif_regex}")
                motif_regex_to_use = default_motif_regex
        except (configparser.NoSectionError, configparser.NoOptionError):
            logger.warning(f"Motif Analysis: '[MotifAnalysis]' section or 'motif_regex_pattern' key not found in {config_path}. Using default regex: {default_motif_regex}")
            motif_regex_to_use = default_motif_regex
    else:
        logger.warning(f"Motif Analysis: config.ini not found at {config_path}. Using default regex: {default_motif_regex}")
    
    logger.debug(f"Final regex pattern for motif analysis: {motif_regex_to_use}")
    # --- End of Motif Regex Configuration ---

    if 'logic_dna_structure_representation' not in df.columns:
        motif_lines.append("\n*Note: `logic_dna_structure_representation` column not found. Skipping motif analysis.*")
        return "\n".join(motif_lines)

    if 'fitness_score' not in df.columns: # Already ensured 'fitness_score_numeric' in main, but good check
        motif_lines.append("\n*Note: `fitness_score` column not found. Skipping motif fitness part of analysis.*")
        overall_avg_fitness = np.nan # Set to NaN if fitness score is missing
    elif 'fitness_score_numeric' not in df.columns: # Should not happen if main pre-processes
        logger.warning("`fitness_score_numeric` not found in DataFrame for motif analysis. This should be pre-calculated.")
        df['fitness_score_numeric'] = pd.to_numeric(df['fitness_score'], errors='coerce')
        overall_avg_fitness = df['fitness_score_numeric'].mean()
    else:
        overall_avg_fitness = df['fitness_score_numeric'].mean()
        if pd.isna(overall_avg_fitness): # If all fitness scores were NaN
             motif_lines.append("\n*Note: No valid numeric fitness scores available for overall average fitness.*")


    compiled_motif_pattern = re.compile(motif_regex_to_use)
    motif_counts: Counter = Counter()

    for structure in df['logic_dna_structure_representation'].dropna():
        if isinstance(structure, str):
            unique_motifs_in_structure = set()
            for match in motif_pattern.finditer(structure):
                motif_name = f"Indicator_{match.group(1)}_{match.group(2)}"
                unique_motifs_in_structure.add(motif_name)
            for motif in unique_motifs_in_structure:
                 motif_counts[motif] += 1
        else:
            logger.debug(f"Skipping non-string structure representation: {structure}")

    if not motif_counts:
        motif_lines.append("\nNo significant indicator motifs found.")
        return "\n".join(motif_lines)

    top_motifs: List[Tuple[str, int]] = motif_counts.most_common(3)

    motif_lines.append("\n| Motif                     | Frequency | Avg. Fitness (Motif) | Avg. Fitness (Overall) |")
    motif_lines.append("|---------------------------|-----------|----------------------|------------------------|")

    motif_regime_correlations: List[str] = ["\n### Motif-Regime Correlation Hints"]
    
    has_ces_column = 'ces_vector_at_evaluation_time' in df.columns

    if not has_ces_column:
        motif_regime_correlations.append("\n*Note: `ces_vector_at_evaluation_time` column not found. Skipping motif-regime correlation analysis.*")


    for motif, frequency in top_motifs:
        motif_containing_df = df[df['logic_dna_structure_representation'].str.contains(motif, na=False, regex=False)]
        
        avg_fitness_motif: float = np.nan
        if 'fitness_score_numeric' in motif_containing_df.columns and not motif_containing_df['fitness_score_numeric'].empty:
            avg_fitness_motif = motif_containing_df['fitness_score_numeric'].mean()
        
        motif_lines.append(f"| {motif:<25} | {frequency:<9} | {avg_fitness_motif:.4f}               | {overall_avg_fitness:.4f}              |")

        # Motif-Regime Correlation Part
        if has_ces_column:
            ces_vector_counts_for_motif: Counter = Counter()
            valid_ces_found_for_motif = False
            for ces_json_str in motif_containing_df['ces_vector_at_evaluation_time'].dropna():
                if isinstance(ces_json_str, str):
                    try:
                        ces_dict = json.loads(ces_json_str)
                        # Serialize the dict to make it hashable for Counter
                        # Sorting items ensures that dicts with same content but different order are treated as same
                        ces_str_repr = json.dumps(ces_dict, sort_keys=True)
                        ces_vector_counts_for_motif[ces_str_repr] += 1
                        valid_ces_found_for_motif = True
                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse CES vector JSON for motif {motif}: {ces_json_str[:100]}...")
                else:
                    logger.debug(f"Skipping non-string CES vector: {ces_json_str}")
            
            correlation_line = f"*   **{motif} (Frequency: {frequency}, Avg. Fitness: {avg_fitness_motif:.4f}):**"
            if valid_ces_found_for_motif and ces_vector_counts_for_motif:
                most_common_ces_str, ces_occurrences = ces_vector_counts_for_motif.most_common(1)[0]
                correlation_line += f"\n    *   Predominantly seen with CES Vector: `{most_common_ces_str}` (CES Occurrences for this motif: {ces_occurrences})"
            elif not valid_ces_found_for_motif:
                correlation_line += f"\n    *   No valid CES vector data found for DNAs containing this motif."
            else: # valid_ces_found_for_motif is True but ces_vector_counts_for_motif is empty (should not happen if valid_ces_found_for_motif is true)
                correlation_line += f"\n    *   CES vector data was present but could not be analyzed for this motif."
            motif_regime_correlations.append(correlation_line)
        else: # has_ces_column is False
             # This case is handled by the initial check, but as a fallback for the loop:
            if motif not in [line_item for line_item in motif_regime_correlations if motif in line_item]: # Avoid duplicate "no data" messages
                 motif_regime_correlations.append(f"*   **{motif}**: No CES vector data available for analysis.")


    motif_lines.append("\n*Note: Fitness comparison provides a basic insight into motif performance.*")
    if has_ces_column and top_motifs: # Only add the section if there was data to analyze
        motif_lines.extend(motif_regime_correlations)
    elif not has_ces_column and top_motifs: # If CES column was missing but we had motifs
         motif_lines.append("\n### Motif-Regime Correlation Hints")
         motif_lines.append("\n*Note: `ces_vector_at_evaluation_time` column not found. Skipping motif-regime correlation analysis.*")

    return "\n".join(motif_lines)

# --- Main Processing and Reporting ---

def process_performance_data(df: pd.DataFrame) -> Dict[str, Any]:
    processed_stats: Dict[str, Any] = {
        'highest_fitness': np.nan,
        'avg_fitness_per_generation': {},
        'total_unique_dna_structures': 0,
        'individual_sharpe_ratios': [],
        'individual_max_drawdowns': [],
        'individual_winning_trade_percentages': [],
        'individual_average_trade_pls': [],
        'dna_with_trades_count': 0
    }

    if 'fitness_score_numeric' in df.columns: # Assumes 'fitness_score_numeric' is pre-calculated
        processed_stats['highest_fitness'] = df['fitness_score_numeric'].max()
        
        valid_fitness_df_local = df.dropna(subset=['fitness_score_numeric'])
        if not valid_fitness_df_local.empty and 'current_generation_evaluated' in df.columns:
            processed_stats['avg_fitness_per_generation'] = valid_fitness_df_local.groupby('current_generation_evaluated')['fitness_score_numeric'].mean().to_dict()
        elif 'current_generation_evaluated' not in df.columns:
            logger.warning("`current_generation_evaluated` column missing for avg_fitness_per_generation.")
        else: # valid_fitness_df_local is empty
             logger.warning("No valid numeric fitness scores for avg_fitness_per_generation.")
    else:
        logger.warning("`fitness_score_numeric` column missing for process_performance_data.")


    if 'logic_dna_structure_representation' in df.columns:
        processed_stats['total_unique_dna_structures'] = df['logic_dna_structure_representation'].nunique()

    for index, row in df.iterrows():
        dna_id = row.get('dna_id', f"Row_{index}")
        metrics_json_str = row.get('performance_metrics')

        if pd.isna(metrics_json_str):
            logger.debug(f"DNA {dna_id}: performance_metrics is missing. Skipping advanced metrics.")
            continue
        try:
            performance_metrics_dict = json.loads(metrics_json_str)
        except json.JSONDecodeError:
            logger.warning(f"DNA {dna_id}: Failed to parse performance_metrics JSON. Skipping. Content: '{metrics_json_str[:100]}...'")
            continue

        equity_curve = performance_metrics_dict.get('equity_curve')
        trade_log = performance_metrics_dict.get('trade_log')

        if not isinstance(equity_curve, list) or not equity_curve:
            logger.debug(f"DNA {dna_id}: equity_curve is missing, not a list, or empty. Skipping Sharpe/Drawdown.")
        else:
            sharpe = calculate_sharpe_ratio(equity_curve)
            if not np.isnan(sharpe): processed_stats['individual_sharpe_ratios'].append(sharpe)
            max_dd = calculate_max_drawdown(equity_curve)
            if not np.isnan(max_dd): processed_stats['individual_max_drawdowns'].append(max_dd)

        if not isinstance(trade_log, list): 
            logger.debug(f"DNA {dna_id}: trade_log is missing or not a list. Skipping trade stats.")
        else:
            trade_stats = calculate_trade_stats(trade_log)
            if trade_stats['total_trades'] > 0:
                processed_stats['dna_with_trades_count'] += 1
                if not np.isnan(trade_stats['winning_trade_percentage']): processed_stats['individual_winning_trade_percentages'].append(trade_stats['winning_trade_percentage'])
                if not np.isnan(trade_stats['average_trade_pl']): processed_stats['individual_average_trade_pls'].append(trade_stats['average_trade_pl'])
            elif not trade_log: 
                logger.debug(f"DNA {dna_id}: No trades in trade_log.")
            else: 
                 logger.debug(f"DNA {dna_id}: trade_log present but resulted in 0 completed trades.")

    processed_stats['average_sharpe_ratio'] = np.mean(processed_stats['individual_sharpe_ratios']) if processed_stats['individual_sharpe_ratios'] else np.nan
    processed_stats['average_max_drawdown'] = np.mean(processed_stats['individual_max_drawdowns']) if processed_stats['individual_max_drawdowns'] else np.nan
    processed_stats['average_winning_trade_percentage'] = np.mean(processed_stats['individual_winning_trade_percentages']) if processed_stats['individual_winning_trade_percentages'] else np.nan
    processed_stats['average_trade_pl_overall'] = np.mean(processed_stats['individual_average_trade_pls']) if processed_stats['individual_average_trade_pls'] else np.nan
    
    return processed_stats

def generate_markdown_report(stats: Dict[str, Any], log_file_name: str, motif_analysis_md: str = "") -> List[str]:
    report_lines: List[str] = []
    report_lines.append(f"# Performance Report for {log_file_name}")
    report_lines.append("\n## Overall Summary Statistics")
    report_lines.append("| Metric                          | Value        |")
    report_lines.append("|---------------------------------|--------------|")
    report_lines.append(f"| Highest Fitness Achieved        | {stats.get('highest_fitness', np.nan):.4f}    |")
    report_lines.append(f"| Overall Sharpe Ratio (Avg)      | {stats.get('average_sharpe_ratio', np.nan):.4f}    |")
    report_lines.append(f"| Maximum Drawdown (Avg)          | {stats.get('average_max_drawdown', np.nan):.2f} %   |")
    report_lines.append(f"| Winning Trade Percentage (Avg)  | {stats.get('average_winning_trade_percentage', np.nan):.2f} %   |")
    report_lines.append(f"| Average Trade Profit/Loss (Avg) | {stats.get('average_trade_pl_overall', np.nan):.4f}    |")
    report_lines.append(f"| Total Unique DNA Structures     | {stats.get('total_unique_dna_structures', 0)}       |")
    report_lines.append(f"| DNAs with Trades Evaluated      | {stats.get('dna_with_trades_count', 0)}       |")

    report_lines.append("\n## Average Fitness Per Generation")
    avg_fitness_gen = stats.get('avg_fitness_per_generation', {})
    if avg_fitness_gen:
        report_lines.append("| Generation | Avg. Fitness |")
        report_lines.append("|------------|--------------|")
        for gen, avg_fitness in sorted(avg_fitness_gen.items()):
            report_lines.append(f"| {str(gen):<10} | {avg_fitness:.4f}       |")
    else:
        report_lines.append("No generation data available or no valid fitness scores for calculation.")
    
    if motif_analysis_md:
        report_lines.append(motif_analysis_md)
        
    return report_lines

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an enhanced performance report from LEE performance logs.")
    parser.add_argument("--log_file_path", type=str, default="performance_log_FULL_V1_2.csv", help="Path to the performance log CSV file.")
    args: argparse.Namespace = parser.parse_args()
    log_file_path_arg: str = args.log_file_path
    actual_log_file_path: str = log_file_path_arg

    if not os.path.isabs(log_file_path_arg) and log_file_path_arg == "performance_log_FULL_V1_2.csv":
        path_in_app: str = os.path.join("/app", log_file_path_arg)
        if os.path.exists(path_in_app) and not os.path.exists(actual_log_file_path) :
            actual_log_file_path = path_in_app
        elif not os.path.exists(actual_log_file_path) and not os.path.exists(path_in_app):
            script_dir = os.path.dirname(os.path.realpath(__file__))
            path_in_script_dir = os.path.join(script_dir, log_file_path_arg)
            if os.path.exists(path_in_script_dir): actual_log_file_path = path_in_script_dir

    logger.info(f"Attempting to read performance log from: {actual_log_file_path}")
    try:
        performance_df: pd.DataFrame = pd.read_csv(actual_log_file_path)
        logger.info(f"Successfully read log file '{os.path.basename(actual_log_file_path)}'. Shape: {performance_df.shape}")
    except FileNotFoundError:
        logger.error(f"Error: Log file not found at {actual_log_file_path}")
        if actual_log_file_path != log_file_path_arg: logger.error(f"Original argument was: {log_file_path_arg}")
        return
    except pd.errors.EmptyDataError:
        logger.error(f"Error: Log file at {actual_log_file_path} is empty.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the log file '{actual_log_file_path}': {e}")
        return

    if performance_df.empty:
        logger.info("The performance log is empty. No statistics to calculate.")
        return

    if 'fitness_score' in performance_df.columns:
        performance_df['fitness_score_numeric'] = pd.to_numeric(performance_df['fitness_score'], errors='coerce')
    else:
        logger.error("Critical: 'fitness_score' column is missing. Many statistics will be unavailable.")
        performance_df['fitness_score_numeric'] = np.nan

    logger.info("Analyzing motif usage...")
    motif_analysis_report_md: str = analyze_motif_usage(performance_df.copy())
    logger.info("Processing general performance data...")
    summary_stats: Dict[str, Any] = process_performance_data(performance_df.copy())
    logger.info("Generating Markdown report...")
    report_markdown_lines: List[str] = generate_markdown_report(summary_stats, os.path.basename(actual_log_file_path), motif_analysis_report_md)
    
    logger.info("\n\n--- Full Performance Report (Markdown) ---")
    for line in report_markdown_lines: logger.info(line)
    logger.info("\n--- End of Report ---")

    # --- HTML Report Generation ---
    logger.info("Generating HTML performance dashboard...")
    # Ensure reports_dir_path is defined in main or passed correctly
    # For this diff, assuming reports_dir_path is "reports" and created in main
    html_report_path = os.path.join("reports", "latest_performance_dashboard.html")
    generate_html_report(summary_stats, performance_df.copy(), html_report_path)
    # --- End of HTML Report Generation ---

if __name__ == "__main__":
    main()


# --- HTML Report Generation Function ---
def generate_html_report(stats: Dict[str, Any], performance_df: pd.DataFrame, html_file_path: str) -> None:
    """
    Generates a simple HTML report with summary statistics and a placeholder for an equity curve chart.

    Args:
        stats (Dict[str, Any]): Dictionary of overall summary statistics.
        performance_df (pd.DataFrame): The main DataFrame to extract a sample equity curve.
        html_file_path (str): Path to save the HTML report.
    """
    logger.info(f"Preparing HTML report content for: {html_file_path}")

    # Extract a sample equity curve
    sample_equity_curve: List[float] = [10000, 10100, 10050, 10200] # Default placeholder
    found_curve = False
    if 'performance_metrics' in performance_df.columns:
        # Iterate a few times to find a curve, not necessarily all rows for performance.
        for _, row_metrics_json_str in performance_df['performance_metrics'].dropna().head(10).items(): 
            if isinstance(row_metrics_json_str, str):
                try:
                    perf_metrics_dict = json.loads(row_metrics_json_str)
                    equity_curve_candidate = perf_metrics_dict.get('equity_curve')
                    if isinstance(equity_curve_candidate, list) and equity_curve_candidate:
                        sample_equity_curve = equity_curve_candidate
                        found_curve = True
                        logger.debug(f"Found an equity curve for HTML report with {len(sample_equity_curve)} points.")
                        break # Use the first valid one found
                except json.JSONDecodeError:
                    logger.debug(f"Could not parse performance_metrics JSON for HTML equity curve: {row_metrics_json_str[:100]}...")
            if found_curve:
                break
        if not found_curve:
            logger.warning("No valid equity curve found in the first 10 performance_metrics entries. Using default placeholder for HTML report.")
    else:
        logger.warning("'performance_metrics' column not found. Using default placeholder equity curve for HTML report.")


    # Build HTML content
    # Using a more robust way to format floats, handling potential 'N/A' if stats keys are missing
    def format_stat(value: Any, precision: int = 4, is_percent: bool = False) -> str:
        if isinstance(value, (int, float)) and not np.isnan(value):
            return f"{value:.{precision}f}{' %' if is_percent else ''}"
        return "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script> <!-- Added Chart.js CDN -->
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        table {{ width: 100%; margin-top: 20px; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #e9e9e9; }}
        .chart-container {{ width:90%; max-width:700px; height:400px; border:1px solid #ccc; margin-top:30px; margin-left:auto; margin-right:auto; background-color: #fff; padding:10px; border-radius: 4px;}}
        .placeholder-text {{ text-align:center; padding-top:150px; color: #888; font-size: 1.2em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Dashboard</h1>
        
        <h2>Overall Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Highest Fitness Achieved</td><td>{format_stat(stats.get('highest_fitness'))}</td></tr>
            <tr><td>Overall Sharpe Ratio (Avg)</td><td>{format_stat(stats.get('average_sharpe_ratio'))}</td></tr>
            <tr><td>Maximum Drawdown (Avg)</td><td>{format_stat(stats.get('average_max_drawdown'), 2, True)}</td></tr>
            <tr><td>Winning Trade Percentage (Avg)</td><td>{format_stat(stats.get('average_winning_trade_percentage'), 2, True)}</td></tr>
            <tr><td>Average Trade Profit/Loss (Avg)</td><td>{format_stat(stats.get('average_trade_pl_overall'))}</td></tr>
            <tr><td>Total Unique DNA Structures</td><td>{stats.get('total_unique_dna_structures', 'N/A')}</td></tr>
            <tr><td>DNAs with Trades Evaluated</td><td>{stats.get('dna_with_trades_count', 'N/A')}</td></tr>
        </table>

        <h2>Sample Equity Curve Trend</h2>
        <div id="equityTrendChartContainer" class="chart-container">
            <canvas id="equityTrendChartCanvas" style="width:100%; height:100%;"></canvas> <!-- Changed div to canvas -->
        </div>
        
    </div>
    <script>
        const equityData = {json.dumps(sample_equity_curve)};
        const ctx = document.getElementById('equityTrendChartCanvas').getContext('2d');
        const labels = Array.from({{ length: equityData.length }}, (_, i) => `Tick ${{i + 1}}`);

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Sample Equity Curve',
                    data: equityData,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'Equity'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Time (Ticks)'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Sample Equity Curve Over Time'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    try:
        # Ensure the reports directory exists (main function already does this, but good practice here too)
        reports_dir = os.path.dirname(html_file_path)
        if reports_dir and not os.path.exists(reports_dir): # Check if reports_dir is not empty string
            os.makedirs(reports_dir, exist_ok=True)
            
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML performance dashboard saved to: {html_file_path}")
    except IOError as e:
        logger.error(f"Error writing HTML report to {html_file_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while generating the HTML report: {e}", exc_info=True)
