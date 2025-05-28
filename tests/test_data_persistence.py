import unittest
import json
import os
import sqlite3 # For type checking errors if needed, not strictly for operations
from typing import Dict, List, Any, Optional

# Attempt to import the class to be tested
try:
    from src.data_persistence import PerformanceLogDatabase
except ImportError:
    PerformanceLogDatabase = None 
    print("CRITICAL: PerformanceLogDatabase could not be imported from src.data_persistence. Tests will fail or be skipped.")

class TestDataPersistence(unittest.TestCase):
    """
    Test suite for the PerformanceLogDatabase class.
    """

    def setUp(self) -> None:
        """
        Set up for test methods. Uses an in-memory SQLite database for each test.
        """
        if PerformanceLogDatabase is None:
            self.skipTest("PerformanceLogDatabase class not available. Skipping tests.")
        
        # Using ":memory:" ensures each test runs against a fresh, isolated database
        # that is discarded when the connection is closed (implicitly or explicitly).
        # The PerformanceLogDatabase __init__ will create the table.
        self.db_instance = PerformanceLogDatabase(db_path=":memory:")
        
        # Verify connection and table creation (optional, but good for sanity)
        conn = self.db_instance._get_connection()
        self.assertIsNotNone(conn, "Database connection should be established for :memory: DB.")
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_log';")
                self.assertIsNotNone(cursor.fetchone(), "'performance_log' table should be created by __init__.")
            except Exception as e:
                self.fail(f"Database setup check failed: {e}")
            finally:
                conn.close()


    def tearDown(self) -> None:
        """
        Clean up after test methods.
        For :memory: databases, closing the connection effectively drops the database.
        The PerformanceLogDatabase class manages connections per method, so explicit
        closing here of a main connection isn't strictly necessary unless we held one open.
        """
        # If self.db_instance held a persistent connection, it would be closed here.
        # Since it doesn't, there's nothing specific to do for :memory: DBs.
        pass

    def test_create_table_idempotent(self) -> None:
        """
        Test if create_table can be called multiple times without error.
        """
        try:
            # setUp already calls create_table once via __init__
            self.db_instance.create_table() # Explicit second call
            self.db_instance.create_table() # Explicit third call
        except Exception as e:
            self.fail(f"Calling create_table() multiple times raised an exception: {e}")

        # Optional: Verify table structure (more detailed)
        conn = self.db_instance._get_connection()
        self.assertIsNotNone(conn)
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(performance_log)")
                columns_info = cursor.fetchall()
                self.assertTrue(len(columns_info) > 0, "Table 'performance_log' should have columns.")
                expected_column_names = [
                    'id', 'dna_id', 'generation_born', 'current_generation_evaluated', 
                    'fitness_score', 'logic_dna_structure_representation', 
                    'performance_metrics_json', 'ces_vector_json', 
                    'active_persona_name', 'timestamp_of_evaluation', 'log_source_file'
                ]
                actual_column_names = [col['name'] for col in columns_info]
                for name in expected_column_names:
                    self.assertIn(name, actual_column_names, f"Expected column '{name}' not found in table schema.")
            finally:
                conn.close()


    def test_insert_and_fetch_all_logs_single_entry(self) -> None:
        sample_log_data: Dict[str, Any] = {
            'dna_id': 'dna_test_001', 'generation_born': 1, 'current_generation_evaluated': 2,
            'fitness_score': 0.75, 'logic_dna_structure_representation': 'RSI_BUY_LOW',
            'performance_metrics': {'profit': 10.5, 'sharpe': 1.2},
            'ces_vector_at_evaluation_time': {'Volatility': 'High', 'Trend': 'Up'},
            'active_persona_name': 'Aggressive', 'timestamp_of_evaluation': '2023-01-01T10:00:00Z'
        }
        source_file = "test_log.csv"

        insert_success = self.db_instance.insert_log_entry(sample_log_data, source_file=source_file)
        self.assertTrue(insert_success, "insert_log_entry should return True on success.")

        fetched_logs = self.db_instance.fetch_all_logs()
        self.assertEqual(len(fetched_logs), 1, "Should fetch one log entry.")

        entry = fetched_logs[0]
        self.assertEqual(entry['dna_id'], sample_log_data['dna_id'])
        self.assertEqual(entry['fitness_score'], sample_log_data['fitness_score'])
        self.assertEqual(entry['log_source_file'], source_file)
        
        # Compare JSON serialized fields
        self.assertEqual(json.loads(entry['performance_metrics_json']), sample_log_data['performance_metrics'])
        self.assertEqual(json.loads(entry['ces_vector_json']), sample_log_data['ces_vector_at_evaluation_time'])
        self.assertEqual(entry['active_persona_name'], sample_log_data['active_persona_name'])

    def test_insert_multiple_entries(self) -> None:
        log1: Dict[str, Any] = {'dna_id': 'dna_m_001', 'fitness_score': 0.5}
        log2: Dict[str, Any] = {'dna_id': 'dna_m_002', 'fitness_score': 0.6}
        
        self.assertTrue(self.db_instance.insert_log_entry(log1, "file1.csv"))
        self.assertTrue(self.db_instance.insert_log_entry(log2, "file2.csv"))

        fetched_logs = self.db_instance.fetch_all_logs()
        self.assertEqual(len(fetched_logs), 2)
        
        # Check if dna_ids are present (simple check for data integrity)
        fetched_dna_ids = {log['dna_id'] for log in fetched_logs}
        self.assertIn('dna_m_001', fetched_dna_ids)
        self.assertIn('dna_m_002', fetched_dna_ids)

    def test_fetch_all_logs_empty_db(self) -> None:
        fetched_logs = self.db_instance.fetch_all_logs()
        self.assertEqual(len(fetched_logs), 0, "fetch_all_logs on empty DB should return an empty list.")

    def test_fetch_top_n_performers(self) -> None:
        logs_to_insert: List[Dict[str, Any]] = [
            {'dna_id': 'dna_top_1', 'fitness_score': 0.7, 'logic_dna_structure_representation': 'S1'},
            {'dna_id': 'dna_top_2', 'fitness_score': 0.9, 'logic_dna_structure_representation': 'S2'},
            {'dna_id': 'dna_top_3', 'fitness_score': 0.5, 'logic_dna_structure_representation': 'S3'},
            {'dna_id': 'dna_top_4', 'fitness_score': 0.95, 'logic_dna_structure_representation': 'S4'},
        ]
        for log in logs_to_insert:
            self.assertTrue(self.db_instance.insert_log_entry(log))

        top_2 = self.db_instance.fetch_top_n_performers(n=2)
        self.assertEqual(len(top_2), 2)
        self.assertEqual(top_2[0]['dna_id'], 'dna_top_4') # Highest fitness
        self.assertEqual(top_2[0]['fitness_score'], 0.95)
        self.assertEqual(top_2[1]['dna_id'], 'dna_top_2') # Second highest
        self.assertEqual(top_2[1]['fitness_score'], 0.9)
        
        # Check content of returned dicts
        for performer in top_2:
            self.assertIn('dna_id', performer)
            self.assertIn('fitness_score', performer)
            self.assertIn('logic_dna_structure_representation', performer)
            self.assertIn('active_persona_name', performer) # This is in the SELECT statement
            self.assertIn('timestamp_of_evaluation', performer) # This is in the SELECT statement

    def test_fetch_top_n_performers_more_than_available(self) -> None:
        log1: Dict[str, Any] = {'dna_id': 'dna_avail_1', 'fitness_score': 0.5}
        log2: Dict[str, Any] = {'dna_id': 'dna_avail_2', 'fitness_score': 0.6}
        self.assertTrue(self.db_instance.insert_log_entry(log1))
        self.assertTrue(self.db_instance.insert_log_entry(log2))

        results = self.db_instance.fetch_top_n_performers(n=5)
        self.assertEqual(len(results), 2, "Should return all available entries if n > count.")
        self.assertEqual(results[0]['dna_id'], 'dna_avail_2') # Higher fitness first

    def test_fetch_top_n_performers_empty_db(self) -> None:
        results = self.db_instance.fetch_top_n_performers(n=5)
        self.assertEqual(len(results), 0, "fetch_top_n_performers on empty DB should return an empty list.")

    def test_json_serialization_in_insert(self) -> None:
        original_perf_metrics = {'profit': 100.0, 'trades': 5, 'win_rate': 0.6}
        original_ces_vector = {'V': 'Low', 'L': 'High', 'T': 'Neutral'}
        log_data: Dict[str, Any] = {
            'dna_id': 'dna_json_test', 'fitness_score': 0.88,
            'performance_metrics': original_perf_metrics,
            'ces_vector_at_evaluation_time': original_ces_vector
        }
        self.assertTrue(self.db_instance.insert_log_entry(log_data))
        
        fetched = self.db_instance.fetch_all_logs()
        self.assertEqual(len(fetched), 1)
        entry = fetched[0]
        
        self.assertIsInstance(entry['performance_metrics_json'], str)
        self.assertIsInstance(entry['ces_vector_json'], str)
        
        loaded_perf_metrics = json.loads(entry['performance_metrics_json'])
        loaded_ces_vector = json.loads(entry['ces_vector_json'])
        
        self.assertEqual(loaded_perf_metrics, original_perf_metrics)
        self.assertEqual(loaded_ces_vector, original_ces_vector)

    def test_insert_log_entry_handles_missing_keys(self) -> None:
        # 'generation_born' and 'ces_vector_at_evaluation_time' are missing
        log_data_missing_keys: Dict[str, Any] = {
            'dna_id': 'dna_missing_keys', 
            'current_generation_evaluated': 5,
            'fitness_score': 0.65, 
            'logic_dna_structure_representation': 'StructureX',
            'performance_metrics': {'profit': 50},
            'active_persona_name': 'Cautious', 
            'timestamp_of_evaluation': '2023-01-02T12:00:00Z'
        }
        
        insert_success = self.db_instance.insert_log_entry(log_data_missing_keys, "missing_keys_test.csv")
        self.assertTrue(insert_success, "Insert should succeed even with some missing optional keys.")

        fetched = self.db_instance.fetch_all_logs()
        self.assertEqual(len(fetched), 1)
        entry = fetched[0]

        self.assertEqual(entry['dna_id'], log_data_missing_keys['dna_id'])
        self.assertIsNone(entry['generation_born'], "Missing 'generation_born' should result in None in DB.")
        
        # ces_vector_json should be a JSON string of an empty dict or None, depending on .get behavior
        # In PerformanceLogDatabase.insert_log_entry:
        # json.dumps(log_data.get('ces_vector_at_evaluation_time', {}))
        # So it will be json.dumps({}) which is '{}'
        self.assertEqual(entry['ces_vector_json'], json.dumps({})) 
        
        # Check that other provided fields are present
        self.assertEqual(entry['current_generation_evaluated'], log_data_missing_keys['current_generation_evaluated'])
        self.assertEqual(json.loads(entry['performance_metrics_json']), log_data_missing_keys['performance_metrics'])


if __name__ == '__main__':
    # Ensure PerformanceLogDatabase is available before trying to run tests
    if PerformanceLogDatabase is None:
        print("Skipping test run: PerformanceLogDatabase class could not be imported.")
    else:
        unittest.main()
