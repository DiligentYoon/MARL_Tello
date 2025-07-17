import ray
import torch
from typing import Dict, Any

# This is a placeholder for your actual environment and agent implementation
# You will need to replace these with your actual classes
from utils.env.apf.apf_env import APFEnv # Assuming this is your environment class
from utils.base.agent.multi_agent import BaseMultiAgent # Assuming this is your agent class

@ray.remote
class RolloutWorker:
    """
    A Ray remote actor responsible for collecting experience from the environment.

    This worker receives model weights from the main driver, runs episodes in its
    own environment instance, and returns the collected data and performance metrics.
    """
    def __init__(self, worker_id: int, env_cfg: Dict[str, Any], agent_cfg: Dict[str, Any]):
        """
        Initializes the worker.

        :param worker_id: A unique ID for the worker.
        :param env_cfg: Configuration dictionary for the environment.
        :param agent_cfg: Configuration dictionary for the agent.
        """
        self.worker_id = worker_id
        self.env = APFEnv(cfg=env_cfg) # Create a new environment instance
        
        # The agent here is a lightweight version, mainly for `act` method.
        # The actual models will be updated from the driver.
        # You might need to adapt this part based on your BaseMultiAgent implementation.
        # For example, you might pass only the policy network to the worker.
        # For now, we assume the worker holds a full agent instance.
        # self.agent = BaseMultiAgent(models={}, device=torch.device("cpu"), cfg=agent_cfg)
        # A placeholder for the policy network, to be updated by the driver
        self.policy_net = None # Replace with your actual policy network class

    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Updates the local policy network with new weights from the driver.
        """
        if self.policy_net is not None:
            self.policy_net.load_state_dict(weights)
        else:
            # Initialize the policy network if it doesn't exist
            # This part needs to be adapted to your actual model architecture
            # from utils.model.model import ActorGaussianNet # Example
            # self.policy_net = ActorGaussianNet(...)
            # self.policy_net.load_state_dict(weights)
            print(f"Worker {self.worker_id}: Policy network initialized.")

    def rollout(self, episode_number: int) -> Dict[str, Any]:
        """
        Runs one episode in the environment to collect a trajectory.

        :param episode_number: The current global episode number (for context).
        :return: A dictionary containing the collected trajectory and performance metrics.
        """
        print(f"[Worker {self.worker_id}] Starting rollout for episode {episode_number}")
        
        trajectory = []
        obs, info = self.env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        episode_length = 0

        while not terminated and not truncated:
            # Ensure policy_net is set
            if self.policy_net is None:
                raise ValueError("Policy network has not been set. Call set_weights first.")

            # Get actions from the local policy
            with torch.no_grad():
                actions, _ = self.policy_net.sample(torch.tensor(obs, dtype=torch.float32))
            
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions.cpu().numpy())
            
            # Store transition
            trajectory.append({
                "obs": obs,
                "actions": actions,
                "rewards": rewards,
                "next_obs": next_obs,
                "terminated": terminated,
                "truncated": truncated
            })
            
            obs = next_obs
            episode_reward += rewards.sum() # Sum rewards across all agents
            episode_length += 1

        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length
        }
        
        print(f"[Worker {self.worker_id}] Finished rollout with reward: {episode_reward}")
        
        return {"trajectory": trajectory, "metrics": metrics, "worker_id": self.worker_id}
