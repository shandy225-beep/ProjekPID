"""
Utility functions untuk proyek ProjekPID
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration dari YAML file
    
    Args:
        config_path: Path ke file config
        
    Returns:
        Dictionary config
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_file: str = "logs/pipeline.log", level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_file: Path ke log file
        level: Logging level
    """
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def format_number(num, decimal=2):
    """Format number with thousand separator"""
    return f"{num:,.{decimal}f}"


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)
