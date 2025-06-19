"""Logging configuration module for FinAgent Pro."""

import logging
import os
from datetime import datetime
from typing import Optional, Any

def setup_logger(
    name: str,
    log_level: int = logging.DEBUG,
    log_to_file: bool = True,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: The name of the logger (usually __name__ of the module)
        log_level: The logging level for the file handler
        log_to_file: Whether to log to a file
        console_level: The logging level for the console handler
    
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Prevent propagation to the root logger
    logger.propagate = False
    
    # Create logs directory if logging to file
    if log_to_file:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f'finagent_{datetime.now().strftime("%Y%m%d")}.log')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with its arguments.
    
    Args:
        logger: The logger instance
        func_name: The name of the function being called
        **kwargs: The function arguments to log
    """
    logger.debug(f"Calling {func_name} with args: {kwargs}")

def log_function_result(logger: logging.Logger, func_name: str, result: Any):
    """
    Log a function's result.
    
    Args:
        logger: The logger instance
        func_name: The name of the function
        result: The result to log
    """
    logger.debug(f"Result from {func_name}: {result}")

def log_error(logger: logging.Logger, error: Exception, context: Optional[str] = None):
    """
    Log an error with optional context.
    
    Args:
        logger: The logger instance
        error: The exception to log
        context: Optional context about where/why the error occurred
    """
    message = f"Error occurred{f' in {context}' if context else ''}: {str(error)}"
    logger.error(message, exc_info=True)
