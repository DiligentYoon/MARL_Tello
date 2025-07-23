import io
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from utils.model.model import ActorGaussianNet, CriticDeterministicNet
from utils.env.apf.apf_env import APFEnv
from utils.env.apf.apf_act_env import APFActEnv
from utils.env.apf.apf_env_cfg import APFEnvCfg
from utils.utils import *



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
    config = load_config("config/sac_cfg.yaml")

    env_cfg = config["env"]
    env = APFActEnv(episode_index=10, cfg=env_cfg)
    print("[INFO] Success to generate Environment")

    policy_cfg = config['model']['actor']
    value_cfg = config['model']['critic']
    policy = ActorGaussianNet(env.cfg.num_obs, env.cfg.num_act, device=torch.device('cuda'), cfg=policy_cfg)
    value = CriticDeterministicNet(env.cfg.num_state + env.cfg.num_act, 1, device=torch.device('cuda'), cfg=value_cfg)
    print(f"[INFO] Success to generate model")

    policy.to(device)

    obs, state, info = env.reset(episode_index=10)

    num_agents = env.num_agent
    num_actions = env.cfg.num_act

    # GIF writers
    belief_writer = imageio.get_writer("belief_update.gif",       mode='I', duration=0.2)
    truth_writer  = imageio.get_writer("truth_path_heading.gif", mode='I', duration=0.2)
    patch_writer  = imageio.get_writer("patches.gif",            mode='I', duration=0.2)

    # Figure 1: belief map + path
    quivers_b = None
    fig_b, ax_b = plt.subplots(figsize=(18,18))
    im_b = ax_b.imshow(env.robot_belief, cmap='gray', vmin=0, vmax=2)
    ax_b.axis('off')
    lines_b = [ax_b.plot([], [], marker='o', markersize=1, label=f"A{i}", color=f"C{i}")[0]
               for i in range(num_agents)]
    ax_b.legend(fontsize=6, loc='upper right')

    # Figure 2: ground truth + path
    fig_t, ax_t = plt.subplots(figsize=(12,12))
    ax_t.imshow(env.ground_truth, cmap='gray', vmin=0, vmax=3)
    ax_t.axis('off')
    lines_t = [ax_t.plot([], [], marker='o', markersize=1, label=f"A{i}", color=f"C{i}")[0]
               for i in range(num_agents)]
    ax_t.legend(fontsize=6, loc='upper right')

    # Figure 3: patches for each agent
    fig_p, axes_p = plt.subplots(1, num_agents, figsize=(3*num_agents,3))
    if num_agents == 1:
        axes_p = [axes_p]
    for ax in axes_p:
        ax.axis('off')

    done = False
    step = 0
    total_step = 256
    
    go_actions = np.ones((num_agents, num_actions), dtype=np.float32)
    go_actions[:, 1] = -0.61
    go_actions[:, 2] = -1


    traj_cells = [[] for _ in range(num_agents)]
    while not done and step < total_step:
        print(f"Steps : {step}/{total_step}")
        next_obs, next_state, reward, terminated, truncated, infos = env.step(go_actions)

        # # 보상 로깅
        # for i, r in enumerate(reward):
        #     print(f"[Step {step}] Agent {i} Reward: {r:.3f}")

        # 위치 & heading 업데이트 및 traj 저장
        for i in range(num_agents):
            new_loc = env.robot_locations[i]
            cell = get_cell_position_from_coords(new_loc, env.belief_info)
            traj_cells[i].append(cell.copy())


        # --- 업데이트 Figure 1: belief map with path ---
        im_b.set_data(env.robot_belief)
        for i in range(num_agents):
            xs = [c[0] for c in traj_cells[i]]
            ys = [c[1] for c in traj_cells[i]]
            lines_b[i].set_data(xs, ys)
        
        if quivers_b:
            quivers_b.remove()
        starts_xy = np.array([traj[-1] for traj in traj_cells])

        # APF 벡터를 가져와 정규화 및 스케일링
        # apf_vectors = env.robot_2d_velocities
        # norms = np.linalg.norm(apf_vectors, axis=1, keepdims=True) + 1e-6
        # # 0으로 나누는 오류 방지
        # normalized_vectors = apf_vectors / norms

        angles = env.angles
        normalized_vectors = np.vstack([np.cos(angles), np.sin(angles)]).transpose()

        arrow_scale = 10.0 # 화살표 길이 (셀 단위)
        scaled_vectors = normalized_vectors * arrow_scale

        # 4. quiver 함수로 새로운 화살표들을 그림
        quivers_b = ax_b.quiver(starts_xy[:, 0], starts_xy[:, 1],          # 화살표 시작점 X, Y
                                  scaled_vectors[:, 0], -scaled_vectors[:, 1], # 화살표 방향 U, V
                                  color='cyan', angles='xy', scale_units='xy', scale=1)

        fig_b.canvas.draw()
        buf_b = io.BytesIO()
        fig_b.savefig(buf_b, format='png', bbox_inches='tight')
        buf_b.seek(0)
        belief_writer.append_data(imageio.imread(buf_b))
        buf_b.close()

        # # --- 업데이트 Figure 2: truth map with path ---
        # for i in range(num_agents):
        #     xs = [c[0] for c in traj_cells[i]]
        #     ys = [c[1] for c in traj_cells[i]]
        #     lines_t[i].set_data(xs, ys)
        # fig_t.canvas.draw()
        # buf_t = io.BytesIO()
        # fig_t.savefig(buf_t, format='png', bbox_inches='tight')
        # buf_t.seek(0)
        # truth_writer.append_data(imageio.imread(buf_t))
        # buf_t.close()

        # # --- 업데이트 Figure 3: patches for each agent ---
        # for i in range(num_agents):
        #     axes_p[i].imshow(env.local_patches[i], cmap='gray', vmin=0, vmax=2)
        #     axes_p[i].set_title(f"A{i} patch", fontsize=8)

        # fig_p.canvas.draw()
        # buf_p = io.BytesIO()
        # fig_p.savefig(buf_p, format='png', bbox_inches='tight')
        # buf_p.seek(0)
        # patch_writer.append_data(imageio.imread(buf_p))
        # buf_p.close()

        step += 1
        done = np.any(terminated) | np.any(truncated)

    # 종료 처리
    belief_writer.close()
    truth_writer.close()
    patch_writer.close()
    plt.close(fig_b)
    plt.close(fig_t)
    plt.close(fig_p)
