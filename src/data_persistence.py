import sqlite3
import os
import json
import logging
import configparser
from typing import List, Dict, Any, Optional

class PerformanceLogDatabase:
    """
    Handles database operations for performance logs using SQLite.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initializes the PerformanceLogDatabase.

        Args:
            db_path (Optional[str]): Path to the SQLite database file.
                                     If None, attempts to read from config.ini,
                                     then falls back to a default path.
        """
        self.logger = logging.getLogger(__name__)
        
        final_db_path: Optional[str] = None
        default_db_path: str = os.path.join("data", "performance_logs.db")

        if db_path:
            final_db_path = db_path
            self.logger.info(f"Database path provided directly: {final_db_path}")
        else:
            config = configparser.ConfigParser()
            # Assuming this file (data_persistence.py) is in src/, config.ini is in project root.
            config_file_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
            
            if os.path.exists(config_file_path):
                config.read(config_file_path)
                try:
                    configured_db_path = config.get('DatabaseSettings', 'db_path', fallback=None)
                    if configured_db_path and configured_db_path.strip():
                        final_db_path = configured_db_path
                        self.logger.info(f"Using database path from config.ini [DatabaseSettings] db_path: {final_db_path}")
                    else:
                        final_db_path = default_db_path
                        self.logger.warning(f"'db_path' in config.ini [DatabaseSettings] is empty or missing. Using default: {final_db_path}")
                except (configparser.NoSectionError, configparser.NoOptionError):
                    final_db_path = default_db_path
                    self.logger.warning(f"'[DatabaseSettings]' section or 'db_path' key not found in {config_file_path}. Using default: {final_db_path}")
            else:
                final_db_path = default_db_path
                self.logger.warning(f"config.ini not found at {config_file_path}. Using default database path: {final_db_path}")
        
        if final_db_path is None : # Should not happen due to default_db_path logic
             final_db_path = default_db_path
             self.logger.error("Critical: final_db_path was None, forced to default. Check logic.")


        self.db_path = final_db_path
        
        # Ensure the directory for the database file exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir): # Check if db_dir is not empty (for relative paths in current dir)
            try:
                os.makedirs(db_dir, exist_ok=True)
                self.logger.info(f"Created database directory: {db_dir}")
            except OSError as e:
                self.logger.error(f"Error creating database directory {db_dir}: {e}", exc_info=True)
                # Potentially raise an error here or handle as critical if DB cannot be created.
                # For now, proceed, and connection will likely fail.

        self.create_table()

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """
        Establishes and returns an SQLite connection.
        Sets row_factory to sqlite3.Row for dictionary-like row access.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database at {self.db_path}: {e}", exc_info=True)
            return None

    def create_table(self) -> None:
        """
        Creates the 'performance_log' table if it doesn't exist.
        """
        sql_create_table = """
        CREATE TABLE IF NOT EXISTS performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dna_id TEXT,
            generation_born INTEGER,
            current_generation_evaluated INTEGER,
            fitness_score REAL,
            logic_dna_structure_representation TEXT,
            performance_metrics_json TEXT,
            ces_vector_json TEXT,
            active_persona_name TEXT,
            timestamp_of_evaluation TEXT,
            log_source_file TEXT
        );
        """
        conn = self._get_connection()
        if conn:
            try:
                with conn: # Context manager handles commit/rollback
                    cursor = conn.cursor()
                    cursor.execute(sql_create_table)
                self.logger.info("Table 'performance_log' checked/created successfully.")
            except sqlite3.Error as e:
                self.logger.error(f"Error creating 'performance_log' table: {e}", exc_info=True)
            finally:
                conn.close()
        else:
            self.logger.error("Cannot create table: database connection failed.")


    def insert_log_entry(self, log_data: Dict[str, Any], source_file: Optional[str] = None) -> bool:
        """
        Inserts a new log entry into the performance_log table.

        Args:
            log_data (Dict[str, Any]): A dictionary containing the log data.
                                       Keys should map to table column names.
            source_file (Optional[str]): The source file of the log entry.

        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        sql_insert = """
        INSERT INTO performance_log (
            dna_id, generation_born, current_generation_evaluated, fitness_score,
            logic_dna_structure_representation, performance_metrics_json,
            ces_vector_json, active_persona_name, timestamp_of_evaluation, log_source_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        # Prepare data, ensuring JSON fields are strings
        performance_metrics = log_data.get('performance_metrics', {})
        ces_vector = log_data.get('ces_vector_at_evaluation_time', {}) # Key from CSV

        data_tuple = (
            log_data.get('dna_id'),
            log_data.get('generation_born'),
            log_data.get('current_generation_evaluated'),
            log_data.get('fitness_score'),
            log_data.get('logic_dna_structure_representation'),
            json.dumps(performance_metrics) if isinstance(performance_metrics, dict) else performance_metrics,
            json.dumps(ces_vector) if isinstance(ces_vector, dict) else ces_vector,
            log_data.get('active_persona_name'),
            log_data.get('timestamp_of_evaluation'),
            source_file
        )

        conn = self._get_connection()
        if conn:
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(sql_insert, data_tuple)
                self.logger.info(f"Log entry for DNA ID {log_data.get('dna_id')} inserted successfully.")
                return True
            except sqlite3.Error as e:
                self.logger.error(f"Error inserting log entry for DNA ID {log_data.get('dna_id')}: {e}", exc_info=True)
                return False
            finally:
                conn.close()
        else:
            self.logger.error("Cannot insert log entry: database connection failed.")
            return False

    def fetch_all_logs(self) -> List[Dict[str, Any]]:
        """
        Fetches all log entries from the performance_log table.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dict represents a row.
                                  Returns an empty list if an error occurs or no data.
        """
        sql_select_all = "SELECT * FROM performance_log;"
        logs: List[Dict[str, Any]] = []
        conn = self._get_connection()
        if conn:
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(sql_select_all)
                    rows = cursor.fetchall()
                    for row in rows:
                        logs.append(dict(row)) # Convert sqlite3.Row to dict
                self.logger.info(f"Fetched {len(logs)} log entries successfully.")
            except sqlite3.Error as e:
                self.logger.error(f"Error fetching all log entries: {e}", exc_info=True)
            finally:
                conn.close()
        else:
            self.logger.error("Cannot fetch logs: database connection failed.")
        return logs

    def fetch_top_n_performers(self, n: int) -> List[Dict[str, Any]]:
        """
        Fetches the top N performers based on fitness_score.

        Args:
            n (int): The number of top performers to fetch.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the top N performers.
                                  Returns an empty list if an error occurs or no data.
        """
        sql_select_top_n = """
        SELECT dna_id, fitness_score, logic_dna_structure_representation, active_persona_name, timestamp_of_evaluation
        FROM performance_log 
        ORDER BY fitness_score DESC 
        LIMIT ?;
        """
        top_performers: List[Dict[str, Any]] = []
        conn = self._get_connection()
        if conn:
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(sql_select_top_n, (n,))
                    rows = cursor.fetchall()
                    for row in rows:
                        top_performers.append(dict(row))
                self.logger.info(f"Fetched top {len(top_performers)} performers successfully.")
            except sqlite3.Error as e:
                self.logger.error(f"Error fetching top {n} performers: {e}", exc_info=True)
            finally:
                conn.close()
        else:
            self.logger.error(f"Cannot fetch top performers: database connection failed.")
        return top_performers

if __name__ == '__main__':
    # Example Usage and Basic Test
    # Ensure basic logging for example run if not configured by main app
    if not logging.getLogger().hasHandlers(): # Check root logger
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger_main = logging.getLogger(__name__)
    logger_main.info("--- Running PerformanceLogDatabase Example ---")

    # Test with default path (will try config.ini, then default 'data/performance_logs.db')
    db_instance = PerformanceLogDatabase() 
    logger_main.info(f"Database instance initialized. DB path: {db_instance.db_path}")

    # Test insertion
    sample_log_1 = {
        'dna_id': 'dna_test_001', 'generation_born': 1, 'current_generation_evaluated': 2,
        'fitness_score': 0.75, 'logic_dna_structure_representation': 'Indicator_RSI_14_BUY_IF_LOW',
        'performance_metrics': {'sharpe': 1.2, 'profit': 150.0}, # This will be JSON stringified
        'ces_vector_at_evaluation_time': {'volatility': 'high', 'trend': 'strong_up'}, # This too
        'active_persona_name': 'AggressiveGrowth', 'timestamp_of_evaluation': '2023-10-26T10:00:00Z'
    }
    db_instance.insert_log_entry(sample_log_1, source_file="example_run.py")

    sample_log_2 = {
        'dna_id': 'dna_test_002', 'generation_born': 1, 'current_generation_evaluated': 2,
        'fitness_score': 0.85, 'logic_dna_structure_representation': 'Indicator_EMA_50_SELL_IF_HIGH',
        'performance_metrics': {'sharpe': 1.5, 'profit': 250.0},
        'ces_vector_at_evaluation_time': {'volatility': 'mid', 'trend': 'sideways'},
        'active_persona_name': 'Conservative', 'timestamp_of_evaluation': '2023-10-26T10:05:00Z'
    }
    db_instance.insert_log_entry(sample_log_2, source_file="example_run.py")
    
    # Test fetching all logs
    all_logs = db_instance.fetch_all_logs()
    logger_main.info(f"\n--- Fetched All Logs ({len(all_logs)}) ---")
    for i, log_entry in enumerate(all_logs):
        logger_main.info(f"Log {i+1}: DNA_ID={log_entry.get('dna_id')}, Fitness={log_entry.get('fitness_score')}")
        # Verify JSON conversion
        if isinstance(log_entry.get('performance_metrics_json'), str):
            logger_main.info(f"  Performance Metrics (JSON string): {log_entry['performance_metrics_json']}")
            try:
                # Attempt to parse it back to dict for verification
                perf_dict = json.loads(log_entry['performance_metrics_json'])
                logger_main.info(f"  Parsed back to dict: {perf_dict}")
            except json.JSONDecodeError:
                logger_main.error("  Failed to parse performance_metrics_json back to dict!")

    # Test fetching top N performers
    top_2 = db_instance.fetch_top_n_performers(n=2)
    logger_main.info(f"\n--- Fetched Top 2 Performers ({len(top_2)}) ---")
    for i, performer in enumerate(top_2):
        logger_main.info(f"Top {i+1}: DNA_ID={performer.get('dna_id')}, Fitness={performer.get('fitness_score')}")

    # Clean up the test database file if you want
    # try:
    #     if os.path.exists(db_instance.db_path):
    #         logger_main.info(f"Cleaning up test database: {db_instance.db_path}")
    #         os.remove(db_instance.db_path)
    # except OSError as e:
    #     logger_main.error(f"Error removing test database file {db_instance.db_path}: {e}")

    logger_main.info("\n--- PerformanceLogDatabase Example Complete ---")
