"""
Logging Utilities for Project Crypto Bot Peter.

This module provides utility functions for setting up and configuring
logging across the application. It aims to centralize logging setup,
allowing for consistent log formatting and output handling (e.g., to
console and/or files) based on configuration.

Key Functions:
- setup_global_logging: Configures the root logger based on an INI file
  or default parameters.
"""
import logging
import logging.config
import configparser
import os

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# More verbose format for DEBUG level, including module and line number
DEBUG_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'

def setup_global_logging(
    config_path: str = 'config.ini',
    default_level: int = logging.INFO,
    log_to_file: bool = True,
    default_log_file_path: str = 'project_peter.log'
) -> None:
    '''
    Configures the root logger for the application.

    Reads logging configuration from a specified .ini file,
    with fallbacks to provided default values.
    '''
    parser = configparser.ConfigParser()
    log_level = default_level
    configured_log_file_path = default_log_file_path
    enable_file_logging = log_to_file
    log_format = DEFAULT_LOG_FORMAT

    logger = logging.getLogger(__name__) # Use a logger specific to this module

    if os.path.exists(config_path):
        try:
            parser.read(config_path)
            if parser.has_section('Logging'):
                level_name_from_config = parser.get('Logging', 'level', fallback=None)
                if level_name_from_config:
                    parsed_level_int = getattr(logging, level_name_from_config.upper(), None)
                    if parsed_level_int is None:
                        logger.warning(
                            f"Invalid log level '{level_name_from_config}' found in {config_path}. "
                            f"Falling back to default level '{logging.getLevelName(default_level)}'."
                        )
                        log_level = default_level  # Fallback
                    else:
                        log_level = parsed_level_int
                else:
                    # 'level' key missing in [Logging] section, use function default
                    log_level = default_level 
                
                # File path
                configured_log_file_path = parser.get('Logging', 'log_file_path', fallback=default_log_file_path)
                
                # Log to file toggle
                log_to_file_str = parser.get('Logging', 'log_to_file', fallback=str(log_to_file))
                if log_to_file_str.lower() == 'false':
                    enable_file_logging = False
                elif log_to_file_str.lower() == 'true':
                    enable_file_logging = True
                # else keep the function's default log_to_file value
                    
                # Log format
                log_format = parser.get('Logging', 'log_format', fallback=DEFAULT_LOG_FORMAT)
            
            else: # No [Logging] section
                logger.info(
                    f"Logging section not found in {config_path}. Using function defaults."
                )
                # log_level, configured_log_file_path, enable_file_logging, log_format
                # will retain their initial values (function defaults).

        except configparser.Error as e:
            logger.warning(f"Error reading logging configuration from {config_path}: {e}. Using function defaults.")
            # Ensure all params are reset to function defaults in case of partial read before error
            log_level = default_level
            configured_log_file_path = default_log_file_path
            enable_file_logging = log_to_file
            log_format = DEFAULT_LOG_FORMAT
            
        except Exception as e: # Catch any other unexpected errors during parsing
            logger.error(f"Unexpected error processing logging config {config_path}: {e}. Using function defaults.")
            log_level = default_level
            configured_log_file_path = default_log_file_path
            enable_file_logging = log_to_file
            log_format = DEFAULT_LOG_FORMAT
    else: # Config file does not exist
        logger.info(f"Configuration file {config_path} not found. Using function defaults.")
        # log_level, configured_log_file_path, enable_file_logging, log_format
        # will retain their initial values (function defaults).


    # Determine which format string to use based on the effective log level
    chosen_format_string = DEBUG_LOG_FORMAT if log_level == logging.DEBUG else log_format

    handlers = []
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(chosen_format_string))
    handlers.append(console_handler)

    # File Handler
    file_handler = None # Initialize to None
    if enable_file_logging:
        try:
            # Ensure directory exists for log file if it's in a subdirectory (Added from my previous impl, good practice)
            log_dir = os.path.dirname(configured_log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                # logging.info(f"Created directory for log file: {log_dir}") # This would use potentially unconfigured logger

            file_handler = logging.FileHandler(configured_log_file_path, mode='a') # Append mode
            # Apply the chosen format string to the file handler as well if it's DEBUG level
            file_handler.setFormatter(logging.Formatter(chosen_format_string))
            handlers.append(file_handler)
        except IOError as e:
            logging.error(f"Could not configure file logging to {configured_log_file_path}: {e}")
        except Exception as e: # Catch any other unexpected errors during file handler setup
            logging.error(f"Unexpected error setting up file logging to {configured_log_file_path}: {e}")


    logging.basicConfig(level=log_level, handlers=handlers, force=True) # force=True to reconfigure if already configured
    
    # Test message
    # Using the module-level logger instance defined at the top.
    file_logging_status = 'Disabled'
    if enable_file_logging:
        # Check if a FileHandler was successfully added to the root logger's handlers
        # This check is a bit indirect; the actual file_handler object from above is more direct.
        if file_handler and any(h is file_handler for h in logging.getLogger().handlers):
            file_logging_status = f'Enabled to {configured_log_file_path}'
        elif file_handler: # File handler was created but perhaps not added to root (should not happen with current logic)
            file_logging_status = f'Attempted for {configured_log_file_path} but check addition to root logger.'
        else: # File handler creation failed (e.g. permission error)
            file_logging_status = 'Failed to enable'


    logger.info(f"Global logging configured. Level: {logging.getLevelName(log_level)}. File logging: {file_logging_status}.")
