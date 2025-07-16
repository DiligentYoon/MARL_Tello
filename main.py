import yaml
import torch
from pathlib import Path
from utils.model import ActorGaussianNet, CriticDeterministicNet

def load_config(config_path: str | Path = 'config.yml') -> dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    # Example usage:
    config = load_config()
    # Print some values to verify
    print("--- Loaded Configuration ---")
    print(f"Number of agents: {config['env']['n_agents']}")
    print(f"Using GPU: {config['train']['use_gpu']}")
    print("--------------------------")

    print("[INFO] Generate Model")
    policy_cfg = config['model']['actor']
    value_cfg = config['model']['critic']
    ActorGaussianNet(policy_cfg['obs_dim'], policy_cfg['act_dim'], device=torch.device('cuda'), cfg=policy_cfg)
    CriticDeterministicNet(value_cfg['obs_dim'], value_cfg['act_dim'], device=torch.device('cuda'), cfg=value_cfg)
    print(f"[INFO] Success to generate model")
