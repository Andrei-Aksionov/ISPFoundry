import os

from omegaconf import OmegaConf

from utils import get_git_root

# Define the path in one single place
CONFIG_PATH = os.getenv("APP_CONFIG_PATH") or (get_git_root() / "src/configs/config.yaml")

# Load it once at the module level
config = OmegaConf.load(CONFIG_PATH)
