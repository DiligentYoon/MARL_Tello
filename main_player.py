import os
import io
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils.agent.masac import MASACAgent
from utils.model.model import ActorGaussianNet, CriticDeterministicNet
from utils.env.apf.apf_env import APFEnv
from utils.utils import get_coords_from_cell_position, get_cell_position_from_coords

class MainPlayer:
    """
    Manages the evaluation of a trained agent without using Ray.
    It loads a checkpoint, runs episodes, reports results, and plots the best episode's trajectory.
    """
    def __init__(self, episode_index: int, cfg: dict, checkpoint_path: str):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.plot_dir = os.path.dirname(checkpoint_path) # Save plot in the same dir as the checkpoint

        # Override config for evaluation
        self.eval_episodes = 1
        self.device = torch.device("cpu") # Evaluation can run on CPU

        # --- Environment and Agent Creation ---
        self.env = APFEnv(episode_index=episode_index, cfg=self.cfg['env'])
        self.num_agents = self.cfg['env']['num_agent']
        obs_dim = self.env.cfg.num_obs
        state_dim = self.env.cfg.num_state
        action_dim = self.env.cfg.num_act

        models = self._create_models(obs_dim, state_dim, action_dim, self.num_agents)
        self.agent = MASACAgent(
            num_agents=self.num_agents,
            models=models,
            device=self.device,
            cfg=self.cfg['agent']
        )
        
        # --- Load Checkpoint ---
        self._load_checkpoint()
        self.agent.policy.eval()
        print("Agent and environment created for evaluation.")



        # --- plot variable ---
        # GIF writers
        self.belief_writer = imageio.get_writer("belief_update.gif",       mode='I', duration=0.2)
        self.truth_writer  = imageio.get_writer("truth_path_heading.gif", mode='I', duration=0.2)
        self.patch_writer  = imageio.get_writer("patches.gif",            mode='I', duration=0.2)



    def _create_models(self, obs_dim, state_dim, action_dim, num_agents) -> dict:
        """Creates the policy and critic models."""
        model_cfg = self.cfg['model']
        policy = ActorGaussianNet(obs_dim, action_dim, self.device, model_cfg['actor'])
        critic1 = CriticDeterministicNet(state_dim + action_dim * num_agents, 1, self.device, model_cfg['critic'])
        critic2 = CriticDeterministicNet(state_dim + action_dim * num_agents, 1, self.device, model_cfg['critic'])
        return {"policy": policy, "critic_1": critic1, "critic_2": critic2}

    def _load_checkpoint(self):
        """Loads model weights directly from a checkpoint file."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.agent.policy.load_state_dict(checkpoint['policy'])
        print(f"--- Policy checkpoint loaded from {self.checkpoint_path} ---")

    def play(self):
        """Main evaluation loop, interacting directly with the environment."""
        print("\n==================================")
        print("===   Evaluation Start         ===")
        print("==================================")

        obs, state, _ = self.env.reset()
        done = False
        total_reward = 0
        length = 0

        # Figure 1: belief map + path
        fig_b, ax_b = plt.subplots(figsize=(5,5))
        im_b = ax_b.imshow(self.env.robot_belief, cmap='gray', vmin=0, vmax=2)
        ax_b.axis('off')
        lines_b = [ax_b.plot([], [], marker='o', markersize=1, label=f"A{i}", color=f"C{i}")[0]
                for i in range(self.num_agents)]
        ax_b.legend(fontsize=6, loc='upper right')

        # Figure 2: ground truth + path
        fig_t, ax_t = plt.subplots(figsize=(5,5))
        ax_t.imshow(self.env.ground_truth, cmap='gray', vmin=0, vmax=3)
        ax_t.axis('off')
        lines_t = [ax_t.plot([], [], marker='o', markersize=1, label=f"A{i}", color=f"C{i}")[0]
                for i in range(self.num_agents)]
        ax_t.legend(fontsize=6, loc='upper right')

        # Figure 3: patches for each agent
        # fig_p, axes_p = plt.subplots(1, self.num_agents, figsize=(3*self.num_agents,3))
        # if self.num_agents == 1:
        #     axes_p = [axes_p]
        # for ax in axes_p:
        #     ax.axis('off')


        go_actions = np.zeros((self.num_agents, self.env.cfg.num_act), dtype=np.float32)
        go_actions[:, 0] = 0.1
        traj_cells = [[] for _ in range(self.num_agents)]
        while not done:
            print(f"Step : ({length}/{self.env.cfg.max_episode_steps})")

            action, _ = self.agent.act(torch.tensor(obs), deterministic=True)
            next_obs, next_state, reward, terminated, truncated, info = self.env.step(action.detach().cpu().numpy())
            # next_obs, next_state, reward, terminated, truncated, info = self.env.step(go_actions)

            for i in range(self.num_agents):
                new_loc = self.env.robot_locations[i]
                cell = get_cell_position_from_coords(new_loc, self.env.belief_info)
                traj_cells[i].append(cell.copy())

            # --- 업데이트 Figure 1: belief map with path ---
            im_b.set_data(self.env.robot_belief)
            for i in range(self.num_agents):
                xs = [c[0] for c in traj_cells[i]]
                ys = [c[1] for c in traj_cells[i]]
                lines_b[i].set_data(xs, ys)
            fig_b.canvas.draw()
            buf_b = io.BytesIO()
            fig_b.savefig(buf_b, format='png', bbox_inches='tight')
            buf_b.seek(0)
            self.belief_writer.append_data(imageio.imread(buf_b))
            buf_b.close()

            # --- 업데이트 Figure 2: truth map with path ---
            for i in range(self.num_agents):
                xs = [c[0] for c in traj_cells[i]]
                ys = [c[1] for c in traj_cells[i]]
                lines_t[i].set_data(xs, ys)
            fig_t.canvas.draw()
            buf_t = io.BytesIO()
            fig_t.savefig(buf_t, format='png', bbox_inches='tight')
            buf_t.seek(0)
            self.truth_writer.append_data(imageio.imread(buf_t))
            buf_t.close()

            # --- 업데이트 Figure 3: patches for each agent ---
            # for i in range(self.num_agents):
            #     axes_p[i].imshow(self.env.local_patches[i], cmap='gray', vmin=0, vmax=2)
            #     axes_p[i].set_title(f"A{i} patch", fontsize=8)

            # fig_p.canvas.draw()
            # buf_p = io.BytesIO()
            # fig_p.savefig(buf_p, format='png', bbox_inches='tight')
            # buf_p.seek(0)
            # self.patch_writer.append_data(imageio.imread(buf_p))
            # buf_p.close()


            done = np.any(terminated) | np.any(truncated)
            obs = next_obs
            total_reward += np.sum(reward)
            length += 1

        print("\n==================================")
        print("===   Evaluation Finished      ===")
        print("==================================")
        print(f"  - Episode Reward: {total_reward:.2f}")
        print(f"  - Episode Length: {length:.2f}")
        print("==================================")

        self.belief_writer.close()
        self.truth_writer.close()
        self.patch_writer.close()
        plt.close(fig_b)
        plt.close(fig_t)
        # plt.close(fig_p)

if __name__ == '__main__':
    # 1. Load configuration
    with open("config/sac_cfg.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 2. Specify the path to the trained model checkpoint
    CHECKPOINT_PATH = os.path.join(os.getcwd(), "results/25-07-20_22-54-37_MARL/checkpoints/agent_48014.pt") # <-- TODO: CHANGE THIS

    # 3. Create and run the player
    if os.path.exists(CHECKPOINT_PATH):
        player = MainPlayer(episode_index=0, cfg=config, checkpoint_path=CHECKPOINT_PATH)
        player.play()
    else:
        print(f"ERROR: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        print("Please update the CHECKPOINT_PATH variable in main_player.py")