import os

CONFIG_NAME = "eph"  # Default config name
config_name = os.getenv("CONFIG") or CONFIG_NAME

# Load corresponding config
from eph_mapf.src.utils.utils import load_config

config = load_config(f"eph_mapf/configs/{config_name}.py")
# Testing
if __name__ == "__main__":
    import rich

    from omegaconf import OmegaConf

    rich.print(OmegaConf.to_yaml(config))
