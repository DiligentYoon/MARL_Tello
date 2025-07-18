import ray
import torch
from typing import Dict, Any

from utils.env.apf.apf_env import APFEnv
from utils.model.model import ActorGaussianNet

@ray.remote
class RolloutWorker:
    """
    A Ray remote actor responsible for collecting experience from the environment.
    It receives policy weights from the MainDriver, runs episodes, and returns
    the collected data.
    """
    def __init__(self, 
                 worker_id: int, 
                 env_cfg: Dict[str, Any], 
                 agent_cfg: Dict[str, Any], 
                 model_cfg: Dict[str, Any]):
        """
        Initializes the worker, its environment, and a local copy of the policy network.
        """
        self.worker_id = worker_id
        self.device = torch.device("cpu") # Workers typically run on CPU

        # --- Environment ---
        self.env = APFEnv(cfg=env_cfg)
        
        # --- Lightweight Agent for Acting ---
        # The worker only needs the policy network to sample actions.
        # The actual learning happens in the MainDriver.
        obs_dim = self.env.observation_space.shape[1]
        action_dim = self.env.action_space.shape[1]
        self.policy_net = ActorGaussianNet(obs_dim, action_dim, self.device, model_cfg['actor'])
        self.policy_net.to(self.device)
        
        print(f"Worker {self.worker_id}: Initialized with env and policy network.")

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Updates the local policy network with new weights from the driver.
        """
        self.policy_net.load_state_dict(weights)

    def rollout(self) -> Dict[str, Any]:
        """
        Runs one full episode in the environment to collect a trajectory.
        """
        trajectory = []
        obs, state, info = self.env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        episode_length = 0

        while not terminated and not truncated:
            # Convert observation to a tensor for the policy network
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

            # Get actions from the local policy
            with torch.no_grad():
                actions, logp = self.policy_net.compute(obs_tensor)
            
            # Step the environment
            next_obs, next_state, rewards, terminated, truncated, infos = self.env.step(actions.cpu().numpy())
            
            # Store the complete transition information
            trajectory.append({
                "obs": obs,
                "state": state,
                "actions": actions.cpu().numpy(),
                "rewards": rewards,
                "next_obs": next_obs,
                "next_state": next_state,
                "terminated": terminated,
                "truncated": truncated
            })
            
            obs = next_obs
            state = next_state
            episode_reward += rewards.sum()
            episode_length += 1

        metrics = {
            f"worker_{self.worker_id}/episode_reward": episode_reward,
            f"worker_{self.worker_id}/episode_length": episode_length,
        }
        
        return {"trajectory": trajectory, "metrics": metrics, "worker_id": self.worker_id}
