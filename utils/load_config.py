import yaml
from pathlib import Path

def load_config(config_path: str):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )
    print(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config