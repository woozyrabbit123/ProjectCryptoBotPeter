import os
import logging
import pytest
from pathlib import Path
from src.utils.logging_utils import setup_global_logging, DEFAULT_LOG_FORMAT
from unittest.mock import MagicMock # Added for mocking

# Clean up logging state before each test
@pytest.fixture(autouse=True)
def reset_logging_state():
    root_logger = logging.getLogger()
    # Close and remove all handlers
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    # Reset level to a known default that won't interfere with most tests
    root_logger.setLevel(logging.WARNING)
    # Ensure no handlers are left
    assert not root_logger.handlers, "Root logger handlers were not fully cleared."
    # Critical: Remove any filters that might have been added by handlers that pytest/caplog might not clean up
    for f in root_logger.filters[:]:
        root_logger.removeFilter(f)
    # Reset manager's loggerDict to ensure no logger instances are cached across tests
    # This helps in making tests more independent, especially if loggers are cached by name.
    logging.Logger.manager.loggerDict.clear()


class TestSetupGlobalLogging:

    def test_setup_global_logging_defaults(self, tmp_path: Path, capsys):
        non_existent_config = tmp_path / "non_existent_config.ini"
        default_log_file = tmp_path / "default.log"

        setup_global_logging(
            config_path=str(non_existent_config),
            default_level=logging.INFO,
            log_to_file=True,
            default_log_file_path=str(default_log_file)
        )

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.INFO
        stream_handler_found = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        assert stream_handler_found, "StreamHandler not found on root logger."
        file_handler = next((h for h in root_logger.handlers if isinstance(h, logging.FileHandler)), None)
        assert file_handler is not None, "FileHandler not found on root logger."
        assert Path(file_handler.baseFilename).resolve() == default_log_file.resolve()
        
        for handler in root_logger.handlers:
            assert isinstance(handler.formatter, logging.Formatter)
            assert handler.formatter._fmt == DEFAULT_LOG_FORMAT

        test_msg = "Defaults test message."
        logging.info(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.err 

        assert default_log_file.exists()
        for handler_to_close in root_logger.handlers[:]:
            if isinstance(handler_to_close, logging.FileHandler):
                handler_to_close.flush(); handler_to_close.close()
        log_content = default_log_file.read_text()
        assert test_msg in log_content
        if default_log_file.exists(): os.remove(default_log_file)


    def test_setup_global_logging_config_level_debug_no_file(self, tmp_path: Path, capsys):
        mock_config_path = tmp_path / "config_debug.ini"
        config_content = """
[Logging]
level = DEBUG
log_to_file = False
"""
        mock_config_path.write_text(config_content)
        setup_global_logging(config_path=str(mock_config_path))

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        assert not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

        test_msg = "This is a debug message."
        logging.debug(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.err
        

    def test_setup_global_logging_config_file_output_warning(self, tmp_path: Path, capsys):
        mock_config_path = tmp_path / "config_warning_file.ini"
        custom_log_file_path = (tmp_path / "custom_test.log").resolve()
        
        config_content = f"""
[Logging]
level = WARNING
log_to_file = True
log_file_path = {str(custom_log_file_path)} 
"""
        mock_config_path.write_text(config_content)
        setup_global_logging(config_path=str(mock_config_path))

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.WARNING
        file_handler = next((h for h in root_logger.handlers if isinstance(h, logging.FileHandler)), None)
        assert file_handler is not None
        assert Path(file_handler.baseFilename).resolve() == custom_log_file_path
        
        test_msg = "This is a warning message for file and console."
        logging.warning(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.err 

        assert custom_log_file_path.exists()
        for handler_to_close in root_logger.handlers[:]: 
            if isinstance(handler_to_close, logging.FileHandler):
                handler_to_close.flush(); handler_to_close.close()
        file_content = custom_log_file_path.read_text()
        assert test_msg in file_content
        if custom_log_file_path.exists(): os.remove(custom_log_file_path)


    def test_setup_global_logging_invalid_config_value_fallback(self, tmp_path: Path, caplog, capsys):
        mock_config_path = tmp_path / "config_invalid.ini"
        expected_log_path = (tmp_path / "test_invalid_fallback.log").resolve()
        config_content = f"""
[Logging]
level = NOTALEVEL 
log_to_file = True
log_file_path = {str(expected_log_path)} 
"""
        mock_config_path.write_text(config_content)
        caplog.set_level(logging.WARNING, logger="src.utils.logging_utils")
        setup_global_logging(config_path=str(mock_config_path), default_level=logging.INFO)

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.INFO 
        assert any("Error reading logging configuration" in record.message and 
                   "NOTALEVEL" in record.message and record.levelname == "WARNING" 
                   for record in caplog.records if record.name == "src.utils.logging_utils")

        test_msg = "Fallback INFO message."
        logging.info(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.err
        if expected_log_path.exists():
            for handler_to_close in root_logger.handlers[:]:
                if isinstance(handler_to_close, logging.FileHandler):
                    handler_to_close.flush(); handler_to_close.close()
            os.remove(expected_log_path)

    def test_setup_global_logging_permission_error_for_logfile(self, tmp_path: Path, caplog, monkeypatch, capsys):
        mock_config_path = tmp_path / "config_perm_error.ini"
        log_file_path_in_config = tmp_path / "permission_denied.log"
        
        config_content = f"""
[Logging]
level = INFO
log_to_file = True
log_file_path = {str(log_file_path_in_config)}
"""
        mock_config_path.write_text(config_content)
        
        monkeypatch.setattr(logging, 'FileHandler', MagicMock(side_effect=PermissionError("Test permission denied")))
        
        caplog.set_level(logging.ERROR, logger="src.utils.logging_utils")
        setup_global_logging(config_path=str(mock_config_path), default_level=logging.INFO)

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.INFO 
        
        assert any("Could not configure file logging" in record.message and
                   "Test permission denied" in record.message and
                   record.levelname == "ERROR"
                   for record in caplog.records if record.name == "src.utils.logging_utils")

        assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        assert not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

        test_msg = "Console logging works despite PermissionError for file."
        logging.info(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.err

    def test_setup_global_logging_malformed_config_ini(self, tmp_path: Path, caplog, capsys):
        malformed_config_path = tmp_path / "malformed_config.ini"
        malformed_config_path.write_text("this is not a valid ini file content--- \n[Malformed")
        
        default_log_file = tmp_path / "malformed_default.log"
        caplog.set_level(logging.WARNING, logger="src.utils.logging_utils")

        setup_global_logging(
            config_path=str(malformed_config_path),
            default_level=logging.INFO,
            default_log_file_path=str(default_log_file)
        )

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.INFO

        assert any("Error reading logging configuration" in record.message and
                   str(malformed_config_path) in record.message
                   for record in caplog.records if record.levelname == "WARNING")

        test_msg = "Malformed config, fallback to INFO."
        logging.info(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.err
        
        assert default_log_file.exists()
        for handler_to_close in root_logger.handlers[:]: 
            if isinstance(handler_to_close, logging.FileHandler):
                handler_to_close.flush(); handler_to_close.close()
        log_content = default_log_file.read_text()
        assert test_msg in log_content
        if default_log_file.exists(): os.remove(default_log_file)


    def test_setup_global_logging_config_missing_logging_section(self, tmp_path: Path, caplog, capsys):
        mock_config_path = tmp_path / "config_no_logging_section.ini"
        config_content = "[General]\nsetting = value"
        mock_config_path.write_text(config_content)
        default_log_file = tmp_path / "missing_section_default.log"

        # setup_global_logging's own INFO message about missing section
        caplog.set_level(logging.INFO, logger="src.utils.logging_utils")

        setup_global_logging(
            config_path=str(mock_config_path),
            default_level=logging.DEBUG, # Use a specific default
            default_log_file_path=str(default_log_file)
        )

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.DEBUG
        
        # Check if the specific "Logging section not found" message was logged
        # This message is logged at INFO level by 'src.utils.logging_utils' in setup_global_logging
        assert any("Logging section not found" in record.message and 
                   record.name == "src.utils.logging_utils" and
                   record.levelname == "INFO" 
                   for record in caplog.records)


        test_msg = "Missing [Logging] section, default DEBUG message."
        logging.debug(test_msg) # Use root logger
        captured = capsys.readouterr() 
        assert test_msg in captured.err
        
        assert default_log_file.exists()
        for handler_to_close in root_logger.handlers[:]: 
            if isinstance(handler_to_close, logging.FileHandler):
                handler_to_close.flush(); handler_to_close.close()
        log_content = default_log_file.read_text()
        assert test_msg in log_content
        if default_log_file.exists(): os.remove(default_log_file)


    def test_setup_global_logging_file_disabled_console_remains(self, tmp_path: Path, capsys):
        mock_config_path = tmp_path / "config_file_disabled.ini"
        config_content = """
[Logging]
level = INFO
log_to_file = False
"""
        mock_config_path.write_text(config_content)
        setup_global_logging(config_path=str(mock_config_path))

        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.INFO
        
        assert len(root_logger.handlers) == 1, "Expected only one handler (StreamHandler)"
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)
        assert not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

        test_msg = "File logging disabled, console INFO message."
        logging.info(test_msg)
        captured = capsys.readouterr()
        assert test_msg in captured.err
