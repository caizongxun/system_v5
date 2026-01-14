import os
import yaml
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = "test/logs") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Logger object
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(config: dict) -> None:
    """
    Create necessary directories from config
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Created/verified directory: {path}")

def print_section(title: str) -> None:
    """
    Print a formatted section header
    
    Args:
        title: Section title
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")
