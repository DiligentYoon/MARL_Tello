import os
import ray
import torch
import yaml
import datetime
import numpy as np
import collections

from torch.utils.tensorboard import SummaryWriter
from utils.buffer.replay_buffer import ReplayBuffer

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
        self.write_interval = self.cfg['agent']['experiment']['write_interval']
        if self.write_interval == 'auto':
            self.write_interval = int(self.timesteps / 10)

        self.checkpoint_interval = self.cfg['agent']['experiment']['checkpoint_interval']
        if self.checkpoint_interval == 'auto':
            self.checkpoint_interval = int(self.timesteps / 10)

        self.cumulative_metrics = {}
        print(f"TensorBoard logs will be saved to: {self.experiment_dir}")

        # --- Environment Info (for model dimensions) ---
        # Create a temporary env to get observation and action dimensions
        temp_env = APFEnv(episode_index=0, cfg=self.cfg['env'])
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
        self.replay_buffer = ReplayBuffer(buffer_size)
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

        # --- Data Logging ---
        self.tracking_data = collections.defaultdict(list)

        self._track_episode_rewards = collections.deque(maxlen=10)
        self._track_episode_lengths = collections.deque(maxlen=10)
        self._track_instantaneous_rewards = collections.deque(maxlen=10)



    def _create_models(self, obs_dim, state_dim, action_dim, num_agents) -> dict:
        """Creates the policy and critic models."""
        model_cfg = self.cfg['model']

        actor_type = model_cfg['actor']['type']
        if actor_type == "Gaussian":
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
        cpu_weights = {k: v.cpu() for k, v in current_weights.items()}
        for worker in self.workers:
            worker.set_weights.remote(cpu_weights)

        # Start the first batch of rollouts
        jobs = [worker.rollout.remote(i) for i, worker in enumerate(self.workers)]
        self.global_episode_count = len(self.workers)

        global_step = 0
        while global_step < self.cfg['train']['timesteps']:
            # Wait for any worker to finish a rollout
            done_ids, jobs = ray.wait(jobs)

            # Process results from all completed workers
            for done_id in done_ids:
                result = ray.get(done_id)
                worker_id = result['worker_id']
                metrics = result['metrics']
                trajectory = result['trajectory']

                # Add collected data to the replay buffer
                for transition in trajectory:
                    self.replay_buffer.push(transition)
                
                # Update Episode Id
                episode_length = metrics[f'episode_length']
                global_step += episode_length

                # Tracking Episode-wise Data from ray Worker
                self._track_data(metrics)

                # --- Centralized Training Step ---
                if len(self.replay_buffer) >= self.cfg['agent']['minimum_buffer_size']:
                    for _ in range(self.cfg['agent']['gradient_steps']):
                        batch = self.replay_buffer.sample(self.cfg['agent']['batch_size'])
                        loss_dict = self.master_agent.update(batch)
                        self._track_data(loss_dict)
                    
                # Update weights to be sent to workers
                current_weights = self.master_agent.get_checkpoint_data()['policy']
                cpu_weights = {k: v.cpu() for k, v in current_weights.items()}

                # --- Logging & Save Checkpoint ---
                # Log
                if global_step > 0 and (global_step - metrics['episode_length']) // self.write_interval < global_step // self.write_interval:
                    self._write_tracking_data(global_step)
                # Checkpoint
                self._save_checkpoint(global_step)

                # --- Launch New Job for the finished worker ---
                self.workers[worker_id].set_weights.remote(cpu_weights)
                new_job = self.workers[worker_id].rollout.remote(self.global_episode_count)
                self.global_episode_count += 1
                jobs.append(new_job)

        print("\n=== Training Finished ===")
        ray.shutdown()



    def _track_data(self, data: dict):
        for key, value in data.items():
            if "episode_reward_team" in key:
                self._track_episode_rewards.append(value)
            elif "episode_length" in key:
                self._track_episode_lengths.append(value)
            elif "instantaneuous_reward_team":
                self._track_instantaneous_rewards.append(value)
            else:
                self.tracking_data[key].append(value)



    def _write_tracking_data(self, global_step: int):
        if len(self._track_episode_rewards) > 0:
            rewards_arr = np.array(self._track_episode_rewards)
            lengths_arr = np.array(self._track_episode_lengths)
            rewards_arr_i = np.array(self._track_instantaneous_rewards)
            
            self.writer.add_scalar("Reward/Total Team reward (mean)", np.mean(rewards_arr), global_step)
            self.writer.add_scalar("Reward/Total Team reward (max)", np.max(rewards_arr), global_step)
            self.writer.add_scalar("Reward/Total Team reward (min)", np.min(rewards_arr), global_step)

            self.writer.add_scalar("Reward/Total Team instantaneous reward (mean)", np.mean(rewards_arr_i), global_step)
            self.writer.add_scalar("Reward/Total Team instantaneous reward (max)", np.max(rewards_arr_i), global_step)
            self.writer.add_scalar("Reward/Total Team instantaneous reward (min)", np.min(rewards_arr_i), global_step)
            
            self.writer.add_scalar("Episode/Total timesteps (mean)", np.mean(lengths_arr), global_step)
            self.writer.add_scalar("Episode/Total timesteps (max)", np.max(lengths_arr), global_step)
            self.writer.add_scalar("Episode/Total timesteps (min)", np.min(lengths_arr), global_step)
        
        for key, values in self.tracking_data.items():
            self.writer.add_scalar(key, np.mean(values), global_step)
        
        self.tracking_data.clear()
        self.writer.flush()



    def _save_checkpoint(self, global_step: int):
        """Saves a checkpoint of the master agent's models."""
        if global_step > 0 and global_step % self.checkpoint_interval == 0:
            filepath = os.path.join(self.experiment_dir, "checkpoints", f"agent_{global_step}.pt")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.master_agent.get_checkpoint_data(), filepath)
            print(f"--- Checkpoint saved at step {global_step} ---")



if __name__ == '__main__':
    with open("config/sac_cfg.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    driver = MainDriver(cfg=config)
    driver.train()

