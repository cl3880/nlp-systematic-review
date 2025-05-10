"""
Utilities for configuring logging in a consistent way across the project.

This module provides functions to set up logging for different scenarios:
1. Basic logging to console and file
2. Model-specific logging (separate log files for different models)
3. Run-specific logging (with timestamps to avoid conflicts)
4. Debug-level logging for troubleshooting
"""
import os
import logging
from datetime import datetime
from src.config import PATHS

def get_timestamp():
    """Get current timestamp in a format suitable for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(name=None, level=logging.INFO, log_dir=None, model_type=None, with_timestamp=True):
    """
    Set up logging with customizable options.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (INFO, DEBUG, etc.)
        log_dir: Directory to store log files
        model_type: Type of model (model-specific logging)
        with_timestamp: Whether to add timestamp to log filename
        
    Returns:
        logger: Configured logger
    """
    if log_dir is None:
        log_dir = PATHS.get("logs_dir", "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    filename_parts = []
    if model_type:
        filename_parts.append(model_type)
    if name and name != "__main__":
        filename_parts.append(name.split(".")[-1])
    if not filename_parts:
        filename_parts.append("main")
    if with_timestamp:
        filename_parts.append(get_timestamp())
    
    log_filename = "_".join(filename_parts) + ".log"
    log_path = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized: {log_path}")
    
    return logger

def setup_per_model_logging(model_type, name=None, level=logging.INFO):
    """
    Set up logging specifically for model runs, with logs saved to model-specific directories.
    
    Args:
        model_type: Type of model (logreg, svm, etc.)
        name: Logger name (typically __name__)
        level: Logging level
        
    Returns:
        logger: Configured logger
    """
    model_log_dir = os.path.join(PATHS.get("logs_dir", "logs"), model_type)
    os.makedirs(model_log_dir, exist_ok=True)
    
    return setup_logging(name=name, level=level, log_dir=model_log_dir, model_type=model_type)

def configure_root_logger(level=logging.INFO):
    """
    Configure the root logger for the project.
    
    Args:
        level: Logging level to set
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    log_dir = PATHS.get("logs_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f"project_{get_timestamp()}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    root_logger.info(f"Root logger initialized: {log_path}")

def set_debug_logging(logger_name=None):
    """
    Temporarily set a specific logger to DEBUG level.
    
    Args:
        logger_name: Name of logger to set to DEBUG (None for root logger)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.debug(f"DEBUG logging enabled for {logger_name or 'root'}") 