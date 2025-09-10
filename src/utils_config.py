# Utility to load YAML config
import yaml

def load_config(config_path):
    """
    Load a YAML configuration file.
    Args:
        config_path (str): Path to YAML config file.
    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
