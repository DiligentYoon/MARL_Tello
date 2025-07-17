import yaml
import torch
import numpy as np
from pathlib import Path
from utils.model.model import ActorGaussianNet, CriticDeterministicNet
from utils.env.apf.apf_env import APFEnv
from utils.env.apf.apf_env_cfg import APFEnvCfg



print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config()

    env_cfg = APFEnvCfg(cfg=config["env"])
    env = APFEnv(episode_index=0, cfg=env_cfg)
    print("[INFO] Success to generate Environment")

    policy_cfg = config['model']['actor']
    value_cfg = config['model']['critic']
    policy = ActorGaussianNet(env.cfg.num_obs, env.cfg.num_act, device=torch.device('cuda'), cfg=policy_cfg)
    value = CriticDeterministicNet(env.cfg.num_state, 1, device=torch.device('cuda'), cfg=value_cfg)
    print(f"[INFO] Success to generate model")

    policy.to(device)

    obs, info = env.reset()

    num_agents = env.num_agent
    num_actions = env.cfg.num_act

    actions, _ = policy.sample(torch.tensor(obs, device=torch.device('cuda'), dtype=torch.float32))

    next_obs, next_state, reward, terminated, truncated, infos = env.step(actions.cpu().detach().numpy())
    print(f"[INFO] Success to Implement Step Methods !")
