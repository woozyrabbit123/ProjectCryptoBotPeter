import os
import pickle
import random
import logging
import pytest
import json # Added for JSON operations
from unittest.mock import patch, MagicMock
import numpy as np # Added
from src.system_orchestrator import SystemOrchestrator

class TestRNGStateManagement:
    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_save_rng_state_success(self, MockMetaParameterMonitor, tmp_path):
        """
        Tests that save_rng_state successfully creates a file
        and the file contains a dictionary with Python's and NumPy's RNG states.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        filepath = tmp_path / "rng_state.pkl"
        
        # Ensure RNGs are in a known, non-default state if possible
        random.seed(42)
        np.random.seed(42)
        _ = random.random() # Advance RNG
        _ = np.random.rand() # Advance RNG

        orchestrator.save_rng_state(filepath)

        assert os.path.exists(filepath)

        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)
        
        assert isinstance(loaded_data, dict)
        assert 'python_random' in loaded_data
        assert 'numpy_random' in loaded_data
        assert isinstance(loaded_data['python_random'], tuple) # Python's state is a tuple
        # NumPy's state is a tuple, the second element of which is the array of numbers.
        assert isinstance(loaded_data['numpy_random'], tuple) 
        # A more specific check for numpy state could be:
        # assert loaded_data['numpy_random'][0] == 'MT19937' # Check bit generator type


    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_rng_state_success(self, MockMetaParameterMonitor, tmp_path):
        """
        Tests that load_rng_state successfully loads both Python's and NumPy's RNG states.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        filepath = tmp_path / "rng_state.pkl"
        
        # Setup and save known states
        random.seed(1337)
        np.random.seed(1337)
        _ = random.random()
        _ = np.random.rand(5) # Generate some numbers to advance numpy state
        
        known_python_state = random.getstate()
        known_numpy_state = np.random.get_state()
        # Further advance numpy RNG to capture a more complex state for saving
        expected_numbers_from_known_numpy_state = np.random.rand(5)


        saved_rng_states = {
            'python_random': known_python_state,
            'numpy_random': known_numpy_state 
        }
        with open(filepath, "wb") as f:
            pickle.dump(saved_rng_states, f)

        # Change current RNG states to something different before loading
        random.seed(999)
        np.random.seed(999)
        _ = random.random()
        _ = np.random.rand(5)
        assert random.getstate() != known_python_state
        # Simple check for numpy state; direct comparison is complex
        # Generate numbers to see if they differ from the expected sequence of the known state
        assert not np.array_equal(np.random.rand(5), expected_numbers_from_known_numpy_state)


        orchestrator.load_rng_state(filepath)
        
        # Verify Python's RNG state
        assert random.getstate() == known_python_state
        
        # Verify NumPy's RNG state by generating numbers
        # This compares the sequence of numbers generated after loading the state
        # with the sequence generated from the original known state.
        restored_numpy_numbers = np.random.rand(5)
        assert np.array_equal(restored_numpy_numbers, expected_numbers_from_known_numpy_state), \
            "NumPy RNG state not restored correctly."

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_rng_state_legacy_format(self, MockMetaParameterMonitor, tmp_path, caplog):
        """
        Tests that load_rng_state can load the old format (just Python's RNG state)
        and logs a warning.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        filepath = tmp_path / "legacy_rng_state.pkl"

        # Save legacy state (just Python's random state)
        random.seed(789)
        _ = random.random()
        legacy_python_state = random.getstate()
        with open(filepath, "wb") as f:
            pickle.dump(legacy_python_state, f)

        # Change current Python state
        random.seed(111)
        assert random.getstate() != legacy_python_state
        
        # Store current NumPy state to ensure it's not affected by legacy load
        np.random.seed(222)
        # Generate a sequence of numbers from this state
        expected_numpy_numbers_after_load = np.random.rand(3) 


        with caplog.at_level(logging.WARNING):
            orchestrator.load_rng_state(filepath)
        
        assert random.getstate() == legacy_python_state
        assert "Loading legacy RNG state (assumed to be Python's random state only)." in caplog.text
        # The INFO message about NumPy state might not be captured if caplog is at WARNING level for this block.
        # The critical part is that the legacy state was loaded and the warning was issued.
        
        # Verify NumPy state is unchanged by re-seeding and checking sequence
        np.random.seed(222) # Re-seed to the state it should have been if untouched by Python-only load
        current_numpy_numbers = np.random.rand(3)
        assert np.array_equal(current_numpy_numbers, expected_numpy_numbers_after_load), \
            "NumPy state was unexpectedly changed by loading a legacy RNG file."


    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_rng_state_partial_dict_python_only(self, MockMetaParameterMonitor, tmp_path, caplog):
        """
        Tests loading a dictionary with only Python's RNG state.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        filepath = tmp_path / "partial_rng_state.pkl"

        random.seed(456)
        _ = random.random()
        python_only_state = random.getstate()
        saved_states = {'python_random': python_only_state}
        with open(filepath, "wb") as f:
            pickle.dump(saved_states, f)
        
        random.seed(1) # Change current state
        np.random.seed(1) # Set numpy state to something known
        # Generate a sequence of numbers from this state
        expected_numpy_numbers_after_load = np.random.rand(3)


        with caplog.at_level(logging.WARNING):
            orchestrator.load_rng_state(filepath)

        assert random.getstate() == python_only_state
        # The primary check is that the warning for the missing key is logged.
        # The successful load of the present key is verified by the state assertion.
        assert "No 'numpy_random' state found in loaded RNG data dictionary." in caplog.text 
        
        # Verify NumPy state is unchanged by re-seeding and checking sequence
        np.random.seed(1) # Re-seed to the state it should have been if untouched by Python-only load
        current_numpy_numbers = np.random.rand(3)
        assert np.array_equal(current_numpy_numbers, expected_numpy_numbers_after_load), \
             "NumPy state was unexpectedly changed when loading python-only state."


    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_rng_state_partial_dict_numpy_only(self, MockMetaParameterMonitor, tmp_path, caplog):
        """
        Tests loading a dictionary with only NumPy's RNG state.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        filepath = tmp_path / "partial_rng_state.pkl"

        np.random.seed(789)
        _ = np.random.rand(3)
        numpy_only_state = np.random.get_state()
        expected_numpy_numbers_after_load = np.random.rand(3) # Sequence after this state

        saved_states = {'numpy_random': numpy_only_state}
        with open(filepath, "wb") as f:
            pickle.dump(saved_states, f)

        random.seed(1) # Set python state to something known
        python_state_before_load = random.getstate()
        
        np.random.seed(2) # Change current numpy state

        with caplog.at_level(logging.WARNING):
            orchestrator.load_rng_state(filepath)

        assert random.getstate() == python_state_before_load, "Python's random state should not have changed."
        # The primary check is that the warning for the missing key is logged.
        # The successful load of the present key is verified by the state assertion.
        assert "No 'python_random' state found in loaded RNG data dictionary." in caplog.text 
        
        current_numpy_numbers = np.random.rand(3)
        assert np.array_equal(current_numpy_numbers, expected_numpy_numbers_after_load), \
            "NumPy state not restored correctly from partial dict."


    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_rng_state_file_not_found(self, MockMetaParameterMonitor, caplog):
        """
        Tests that load_rng_state handles a non-existent file gracefully.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        filepath = "non_existent_file.pkl"

        with caplog.at_level(logging.WARNING):
            orchestrator.load_rng_state(filepath)
        
        assert f"RNG state file not found at {filepath}. Skipping load." in caplog.text
        # Assert that the program continues execution (no exception raised)

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_rng_state_corrupted_file(self, MockMetaParameterMonitor, tmp_path, caplog):
        """
        Tests that load_rng_state handles a corrupted RNG state file gracefully.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        filepath = tmp_path / "corrupted_rng_state.pkl"

        # Create a corrupted file
        with open(filepath, "wb") as f:
            f.write(b"corrupted_data")

        with caplog.at_level(logging.ERROR): # Assuming it logs an error for corrupted files
            orchestrator.load_rng_state(filepath)
        
        assert f"Error unpickling RNG state from {filepath}: pickle data was truncated" in caplog.text
        # Assert that the program continues execution (no exception raised)

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_rng_state_truly_corrupted_unpickleable_file(self, MockMetaParameterMonitor, tmp_path, caplog):
        """
        Tests that load_rng_state handles a file with non-pickle binary content gracefully
        and does not change existing RNG states.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        
        # Get initial RNG states
        random.seed(123) # Set a known state
        np.random.seed(123) # Set a known state
        initial_python_state = random.getstate()
        initial_numpy_state_tuple = np.random.get_state() # np.random.get_state() returns a tuple

        corrupted_rng_file = tmp_path / "corrupted_text.rng"
        corrupted_rng_file.write_text("This is definitely not a pickle file.")

        caplog.set_level(logging.ERROR, logger="src.system_orchestrator")
        orchestrator.load_rng_state(str(corrupted_rng_file))

        # Verify error message
        assert "Error unpickling RNG state from" in caplog.text
        assert str(corrupted_rng_file) in caplog.text
        
        # Verify RNG states remain unchanged
        assert random.getstate() == initial_python_state, "Python's RNG state was unexpectedly changed."
        
        # Compare NumPy state elements individually if direct tuple comparison is flaky
        current_numpy_state_tuple = np.random.get_state()
        assert current_numpy_state_tuple[0] == initial_numpy_state_tuple[0], "NumPy RNG state (type) was unexpectedly changed."
        assert np.array_equal(current_numpy_state_tuple[1], initial_numpy_state_tuple[1]), "NumPy RNG state (keys) was unexpectedly changed."
        assert current_numpy_state_tuple[2] == initial_numpy_state_tuple[2], "NumPy RNG state (pos) was unexpectedly changed."
        assert current_numpy_state_tuple[3] == initial_numpy_state_tuple[3], "NumPy RNG state (has_gauss) was unexpectedly changed."
        # The cached_gaussian element can sometimes be an array, handle that.
        if isinstance(current_numpy_state_tuple[4], np.ndarray) and isinstance(initial_numpy_state_tuple[4], np.ndarray):
            assert np.array_equal(current_numpy_state_tuple[4], initial_numpy_state_tuple[4]), "NumPy RNG state (cached_gaussian) was unexpectedly changed."
        else:
            assert current_numpy_state_tuple[4] == initial_numpy_state_tuple[4], "NumPy RNG state (cached_gaussian) was unexpectedly changed."


    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_save_rng_state_raises_error_if_dir_not_exists(self, MockMetaParameterMonitor, tmp_path):
        """
        Tests that save_rng_state raises FileNotFoundError if the parent directory for the save path does not exist.
        This confirms current behavior that os.makedirs is not implicitly called.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        
        save_path = tmp_path / "non_existent_dir" / "state.rng"
        
        # Standard open() in 'wb' mode does not create parent directories.
        # We expect FileNotFoundError (or possibly IOError, depending on Python version/OS nuances,
        # but FileNotFoundError is more specific and common for this case).
        with pytest.raises(FileNotFoundError):
            orchestrator.save_rng_state(str(save_path))


# Helper for valid lee_params (already defined, ensure it's available or re-add if needed)
# def get_valid_lee_params() -> dict: ...

class TestConfigLoading:
    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_missing_lee_params_section(self, MockMetaParameterMonitor, tmp_path):
        """
        Tests that _load_config raises RuntimeError for a non-existent config file.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        non_existent_filepath = "non_existent_config.json"

        with pytest.raises(RuntimeError) as excinfo:
            orchestrator._load_config(non_existent_filepath)
        
        assert f"Configuration file not found at path: {non_existent_filepath}" in str(excinfo.value)

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_json_decode_error(self, MockMetaParameterMonitor, tmp_path):
        """
        Tests that _load_config raises RuntimeError for a corrupted JSON config file.
        """
        MockMetaParameterMonitor.return_value = MagicMock()
        orchestrator = SystemOrchestrator(None, None, None)
        corrupted_filepath = tmp_path / "corrupted_config.json"

        with open(corrupted_filepath, "w") as f:
            f.write("this is not valid json")

        with pytest.raises(RuntimeError) as excinfo:
            orchestrator._load_config(corrupted_filepath)
            
        assert f"Error decoding JSON configuration file at path: {corrupted_filepath}. Details: Expecting value: line 1 column 1 (char 0)" in str(excinfo.value)

# Helper for valid lee_params
def get_valid_lee_params() -> dict:
    return {
        "population_size": 10, 
        "max_depth": 5,
        "mutation_rate_parametric": 0.1,
        "mutation_rate_structural": 0.1,
        "crossover_rate": 0.5,
        "elitism_percentage": 0.1,
        "random_injection_percentage": 0.1,
        "max_nodes": 100,
        "complexity_weights": {"nodes": 1, "depth": 1}
    }

class TestConfigValidation:
    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_missing_lee_params_section(self, MockMetaParameterMonitor, tmp_path): # Renamed test
        MockMetaParameterMonitor.return_value = MagicMock()
        # Config data deliberately missing the entire 'lee_params' section
        config_data = { "personas": {"default": {}}, "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)
        
        with pytest.raises(ValueError, match="'lee_params' section is missing or not a dictionary"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_missing_lee_population_size(self, MockMetaParameterMonitor, tmp_path): # Renamed
        MockMetaParameterMonitor.return_value = MagicMock()
        lee_params = get_valid_lee_params()
        del lee_params["population_size"] # Specifically remove population_size
        config_data = {"lee_params": lee_params, "personas": {"default": {}}, "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(KeyError, match="is missing required key: 'population_size'"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_type_lee_population_size(self, MockMetaParameterMonitor, tmp_path): # Renamed
        MockMetaParameterMonitor.return_value = MagicMock()
        lee_params = get_valid_lee_params()
        lee_params["population_size"] = "not_an_int" # Invalid type
        config_data = {"lee_params": lee_params, "personas": {"default": {}}, "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match=r"must be type int\. Found: str"): # Updated match
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_value_lee_population_size(self, MockMetaParameterMonitor, tmp_path): # Renamed
        MockMetaParameterMonitor.return_value = MagicMock()
        lee_params = get_valid_lee_params()
        lee_params["population_size"] = 0 # Invalid value
        config_data = {"lee_params": lee_params, "personas": {"default": {}}, "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match=r"value 0 is not valid"): # Updated match
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_type_rng_load_path(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {"lee_params": get_valid_lee_params(), "personas": {"default": {}}, "rng_state_load_path": 123, "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="'rng_state_load_path' must be a non-empty string if provided"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_rng_load_path_not_found(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        non_existent_rng_file = tmp_path / "non_existent_rng.pkl"
        config_data = {"lee_params": get_valid_lee_params(), "personas": {"default": {}}, "rng_state_load_path": str(non_existent_rng_file), "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(FileNotFoundError, match="'rng_state_load_path' file not found"):
            SystemOrchestrator(config_file_path=str(config_filepath))
    
    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_type_rng_save_path(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {"lee_params": get_valid_lee_params(), "personas": {"default": {}}, "rng_state_save_path": 123}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="'rng_state_save_path' must be a non-empty string if provided"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('os.access', return_value=False) # Mock os.access to simulate not writable
    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_rng_save_path_not_writable(self, MockMetaParameterMonitor, mock_os_access, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        dummy_load_file = tmp_path / "dummy_rng_load.pkl" # Create a valid load file
        with open(dummy_load_file, "wb") as f: pickle.dump(random.getstate(),f)

        config_data = {
            "lee_params": get_valid_lee_params(), 
            "personas": {"default": {}},
            "rng_state_load_path": str(dummy_load_file), 
            "rng_state_save_path": "unwritable_dir/rng_save.pkl"
        }
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(IOError, match="Directory for 'rng_state_save_path' .* is not writable"):
            SystemOrchestrator(config_file_path=str(config_filepath))
        mock_os_access.assert_called_with('unwritable_dir', os.W_OK)


    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_type_priming_generations(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {"lee_params": get_valid_lee_params(), "personas": {"default": {}}, "priming_generations": "not_an_int", "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="'priming_generations' must be an integer >= 0"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_value_priming_generations(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {"lee_params": get_valid_lee_params(), "personas": {"default": {}}, "priming_generations": -1, "rng_state_save_path": str(tmp_path / "save.pkl")}
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="'priming_generations' must be an integer >= 0"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_missing_personas_section(self, MockMetaParameterMonitor, tmp_path): # Renamed
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {"lee_params": get_valid_lee_params(), "rng_state_save_path": str(tmp_path / "save.pkl")} # Missing personas
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)
        
        with pytest.raises(ValueError, match="'personas' section is missing or not a dictionary"):
            SystemOrchestrator(config_file_path=str(config_filepath))
            
    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_valid_minimal(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {
            "lee_params": get_valid_lee_params(),
            "personas": {"default": {}},
            "rng_state_save_path": str(tmp_path / "save.pkl") # Ensure save path is writable for this test
        }
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)
        
        try:
            SystemOrchestrator(config_file_path=str(config_filepath))
        except (ValueError, FileNotFoundError, IOError, KeyError) as e: # Added KeyError
            pytest.fail(f"Minimal valid config raised an unexpected error: {e}")
            
    @patch('src.system_orchestrator.MetaParameterMonitor')
    @patch('os.access', return_value=True) # Ensure save path is considered writable
    def test_load_config_valid_full(self, mock_os_access, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        
        dummy_load_file = tmp_path / "dummy_rng_load.pkl"
        with open(dummy_load_file, "wb") as f: pickle.dump(random.getstate(),f)

        config_data = get_valid_lee_params() # Start with valid lee_params
        config_data = { # Build the full config
            "lee_params": get_valid_lee_params(),
            "personas": {"default": {"fitness_weights": {"profit": 0.5, "sharpe": 0.5}}},
            "initial_active_persona": "default",
            "performance_log_path": "test_perf.log",
            "rng_state_load_path": str(dummy_load_file),
            "rng_state_save_path": str(tmp_path / "test_save.pkl"),
            "priming_generations": 5,
            "conflict_resolver_config": {}
        }
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)
        
        # Should not raise any validation errors
        try:
            SystemOrchestrator(config_file_path=str(config_filepath))
        except (ValueError, FileNotFoundError, IOError) as e:
            pytest.fail(f"Full valid config raised an unexpected error: {e}")

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_missing_lee_params_mutation_rate_parametric(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        lee_params = get_valid_lee_params()
        del lee_params["mutation_rate_parametric"]
        config_data = {
            "lee_params": lee_params,
            "personas": {"default": {}},
            "rng_state_save_path": str(tmp_path / "save.pkl")
        }
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(KeyError, match="'lee_params' is missing required key: 'mutation_rate_parametric'"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_type_lee_params_mutation_rate_structural(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        lee_params = get_valid_lee_params()
        lee_params["mutation_rate_structural"] = "not_a_float"
        config_data = {
            "lee_params": lee_params,
            "personas": {"default": {}},
            "rng_state_save_path": str(tmp_path / "save.pkl")
        }
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match=r"'lee_params.mutation_rate_structural' must be type float\. Found: str"):
            SystemOrchestrator(config_file_path=str(config_filepath))

    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_type_priming_generations_string(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {
            "lee_params": get_valid_lee_params(),
            "personas": {"default": {}},
            "priming_generations": "not_an_int_either", # Already covered by invalid_type_priming_generations
            "rng_state_save_path": str(tmp_path / "save.pkl")
        }
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="'priming_generations' must be an integer >= 0"):
            SystemOrchestrator(config_file_path=str(config_filepath))
            
    @patch('src.system_orchestrator.MetaParameterMonitor')
    def test_load_config_invalid_type_performance_log_path(self, MockMetaParameterMonitor, tmp_path):
        MockMetaParameterMonitor.return_value = MagicMock()
        config_data = {
            "lee_params": get_valid_lee_params(),
            "personas": {"default": {}},
            "performance_log_path": 12345, # Invalid type
            "rng_state_save_path": str(tmp_path / "save.pkl")
        }
        config_filepath = tmp_path / "config.json"
        with open(config_filepath, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="'performance_log_path' must be a non-empty string if provided"):
            SystemOrchestrator(config_file_path=str(config_filepath))
