import os
import ray
import torch
import yaml
import datetime
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from utils.runner.rollout_worker import RolloutWorker
from utils.agent.masac import MASACAgent
from utils.model.model import ActorGaussianNet, CriticDeterministicNet
from utils.env.apf.apf_env import APFEnv # Assuming this is the environment to get dims

class MainDriver:
    """
    The main orchestrator for the training process.
    It manages worker creation, data collection, centralized training,
    logging, and checkpointing.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.timesteps = self.cfg["train"]["timesteps"]
        self.start_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        
        ray.init(num_cpus=self.cfg['ray']['num_cpus'])
        print(f"Ray initialized with {self.cfg['ray']['num_cpus']} CPUs.")

        self.device = torch.device(self.cfg['env']['device'])
        
        # --- Experiment Directory and Logging ---
        self.experiment_dir = os.path.join("results", f"{self.start_time}_{self.cfg['agent']['experiment']['directory']}")
        self.writer = SummaryWriter(log_dir=self.experiment_dir)
        if self.cfg['agent']['experiment']['write_interval'] == 'auto':
            self.write_interval = int(self.timesteps / 10)
        if self.cfg['agent']['experiment']['checkpoint_interval'] == 'auto':
            self.checkpoint_interval = int(self.timesteps / 10)
        print(f"TensorBoard logs will be saved to: {self.experiment_dir}")

        # --- Environment Info (for model dimensions) ---
        # Create a temporary env to get observation and action dimensions
        temp_env = APFEnv(cfg=self.cfg['env'])
        obs_dim = temp_env.cfg.num_obs
        state_dim = temp_env.cfg.num_state
        action_dim = temp_env.cfg.num_act
        num_agents = self.cfg['env']['num_agent']
        del temp_env

        # --- Centralized Components ---
        # 1. Master Agent (holds the master networks and optimizers)
        models = self._create_models(obs_dim, state_dim, action_dim, num_agents)
        self.master_agent = MASACAgent(
            num_agents=num_agents,
            models=models,
            device=self.device,
            cfg=self.cfg['agent']
        )
        print("Master MASACAgent created.")

        # 2. Replay Buffer
        buffer_size = self.cfg['agent']['buffer']['replay_size']
        self.replay_buffer = deque(maxlen=buffer_size)
        print(f"Replay buffer created with max size {buffer_size}.")

        # --- Worker Creation for Parallel Working ---
        self.workers = [
            RolloutWorker.remote(
                worker_id=i, 
                env_cfg=self.cfg['env'], 
                agent_cfg=self.cfg['agent'],
                model_cfg=self.cfg['model'] # Pass model config to workers
            )
            for i in range(self.cfg['ray']['num_workers'])
        ]
        print(f"{self.cfg['ray']['num_workers']} RolloutWorkers created.")



    def _create_models(self, obs_dim, state_dim, action_dim, num_agents) -> dict:
        """Creates the policy and critic models."""
        model_cfg = self.cfg['model']

        model_cls = model_cfg["class"]

        if model_cls in ["PPO", "SAC"]:
            policy = ActorGaussianNet(obs_dim, action_dim, self.device, model_cfg['actor'])
        else:
            ValueError("[INFO] TODO : we should construct the deterministic policy for MADDPG ...")
        
        # Centralized critic input dimension: state + agent actions
        critic_input_dim = state_dim + action_dim
        
        critic1 = CriticDeterministicNet(critic_input_dim, 1, self.device, model_cfg['critic'])
        critic2 = CriticDeterministicNet(critic_input_dim, 1, self.device, model_cfg['critic'])
        
        return {"policy": policy, "critic_1": critic1, "critic_2": critic2}


    def train(self):
        """Main training loop."""
        print("=== Training Start ===")
        
        current_weights = self.master_agent.get_checkpoint_data()['policy']
        
        # Broadcast initial weights to all workers
        for worker in self.workers:
            worker.set_weights.remote(current_weights)

        # Start the first batch of rollouts
        jobs = [worker.rollout.remote() for worker in self.workers]

        global_step = 0
        while global_step < self.cfg['train']['timesteps']:
            done_ids, jobs = ray.wait(jobs)
            result = ray.get(done_ids[0])

            # --- Process Worker Results ---
            worker_id = result['worker_id']
            metrics = result['metrics']
            trajectory = result['trajectory']

            self.replay_buffer.extend(trajectory)
            episode_length = metrics[f'worker_{worker_id}/episode_length']
            global_step += episode_length

            self._log_metrics(metrics, global_step)

            # --- Centralized Training Step ---
            if len(self.replay_buffer) >= self.cfg['agent']['batch_size']:
                for _ in range(self.cfg['agent']['gradient_steps']):
                    # This part needs a proper ReplayBuffer implementation
                    # For now, we'll skip the actual batch sampling and update
                    pass
                
                # loss_dict = self.master_agent.update(batch)
                # self._log_metrics(loss_dict, global_step)
                
                # Update weights to be sent to workers
                current_weights = self.master_agent.get_checkpoint_data()['policy']

            # --- Checkpointing ---
            self._save_checkpoint(global_step)

            # --- Launch New Job ---
            # Relaunch the job on the worker that just finished
            # and send it the latest weights
            new_job = self.workers[worker_id].rollout.remote()
            self.workers[worker_id].set_weights.remote(current_weights)
            jobs.append(new_job)

        print("\n=== Training Finished ===")
        ray.shutdown()

    def _log_metrics(self, metrics: dict, global_step: int):
        """Logs metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, global_step)
        self.writer.flush()

    def _save_checkpoint(self, global_step: int):
        """Saves a checkpoint of the master agent's models."""
        checkpoint_interval = self.cfg['agent']['experiment'].get('checkpoint_interval', 50000)
        if global_step > 0 and global_step % checkpoint_interval < self.cfg['agent']['gradient_steps']:
            filepath = os.path.join(self.experiment_dir, "checkpoints", f"step_{global_step}.pt")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.master_agent.get_checkpoint_data(), filepath)
            print(f"--- Checkpoint saved at step {global_step} ---")



if __name__ == '__main__':
    with open("sac_cfg.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    driver = MainDriver(cfg=config)
    driver.train()

