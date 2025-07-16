import yaml
import torch
from pathlib import Path
from utils.model import ActorGaussianNet, CriticDeterministicNet
from utils.apf_env import APFEnv
from utils.env_apf_cfg import APFEnvCfg

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

    env_cfg = APFEnvCfg(cfg=config["env"])
    env = APFEnv(episode_index=0, cfg=env_cfg)
    print("[INFO] Success to generate Environment")

    policy_cfg = config['model']['actor']
    value_cfg = config['model']['critic']
    ActorGaussianNet(env.cfg.num_obs, env.cfg.num_act, device=torch.device('cuda'), cfg=policy_cfg)
    CriticDeterministicNet(env.cfg.num_state, 1, device=torch.device('cuda'), cfg=value_cfg)
    print(f"[INFO] Success to generate model")


    
