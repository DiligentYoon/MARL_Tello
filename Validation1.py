import os
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from utils.env   import Env
from utils.agent import Agent
from parameter   import *
from utils.utils import get_cell_position_from_coords

def main():
    # 환경·에이전트 초기화
    env = Env(episode_index=0, fov=FOV, sensor_range=SENSOR_RANGE, plot=False)
    agents = []
    for i in range(N_AGENTS):
        a = Agent(
            id=i, policy_net=None, env=env,
            fov=FOV, heading=env.angles[i],
            sensor_range=SENSOR_RANGE, device='cpu', plot=False
        )
        a.prev_velocity = 0.0 
        a.prev_yaw_rate = 0.0 
        agents.append(a)

    env.robot_locations = [[0.2, 0.5], [0.2, 0.6], [0.2, 0.7], [0.2, 0.8]]
    env.angles = [0, 0, 0, 0]

    # 궤적 저장용
    traj_cells = [[] for _ in range(N_AGENTS)]

    # GIF writers
    belief_writer = imageio.get_writer("belief_update.gif",       mode='I', duration=0.2)
    truth_writer  = imageio.get_writer("truth_path_heading.gif", mode='I', duration=0.2)
    patch_writer  = imageio.get_writer("patches.gif",            mode='I', duration=0.2)

    # Figure 1: belief map + path
    fig_b, ax_b = plt.subplots(figsize=(5,5))
    im_b = ax_b.imshow(env.robot_belief, cmap='gray', vmin=0, vmax=2)
    ax_b.axis('off')
    lines_b = [ax_b.plot([], [], marker='o', markersize=1, label=f"A{i}", color=f"C{i}")[0]
               for i in range(N_AGENTS)]
    ax_b.legend(fontsize=6, loc='upper right')

    # Figure 2: ground truth + path
    fig_t, ax_t = plt.subplots(figsize=(5,5))
    ax_t.imshow(env.ground_truth, cmap='gray', vmin=0, vmax=3)
    ax_t.axis('off')
    lines_t = [ax_t.plot([], [], marker='o', markersize=1, label=f"A{i}", color=f"C{i}")[0]
               for i in range(N_AGENTS)]
    ax_t.legend(fontsize=6, loc='upper right')

    # Figure 3: patches for each agent
    fig_p, axes_p = plt.subplots(1, N_AGENTS, figsize=(3*N_AGENTS,3))
    if N_AGENTS == 1:
        axes_p = [axes_p]
    for ax in axes_p:
        ax.axis('off')

    done = False
    step = 0
    while not done and step < 100:
        # 관측 & 랜덤 액션
        observations = [a.get_observation(agents) for a in agents]
        for i, obs in enumerate(observations):
            print(
                f"[Step {step}] Agent {i} Observation: "
                + " ".join(f"{v:.3f}" for v in obs.cpu().numpy().flatten())
            )

        # 1) 랜덤 액션 생성 & env.step()
        actions = []
        for i, a in enumerate(agents):
            # v        = np.random.uniform(0, MAX_VELOCITY)
            # yaw_rate = np.random.uniform(-MAX_YAW_RATE, MAX_YAW_RATE)
            v        = 0.5
            yaw_rate = 0
            a.prev_yaw_rate = yaw_rate
            a.prev_velocity = v
            actions.append((v, yaw_rate))
            print(f"[Step {step}] Agent {i} Action: v={v:.2f}, yaw={yaw_rate:.2f}")

        # 환경 갱신
        _, reward_list, done, _ = env.step(actions)

        # 보상 로깅
        for i, r in enumerate(reward_list):
            print(f"[Step {step}] Agent {i} Reward: {r:.3f}")

        # 위치 & heading 업데이트 및 traj 저장
        for i, a in enumerate(agents):
            new_loc = env.robot_locations[i]
            a.location = new_loc
            a.heading  = env.angles[i]
            cell = get_cell_position_from_coords(new_loc, env.belief_info)
            traj_cells[i].append(cell.copy())
            # 새로: agent.path 업데이트 (예: compute_apf_patch 등에서)
            # a.path = compute_patch_somehow(a)  

        # --- 업데이트 Figure 1: belief map with path ---
        im_b.set_data(env.robot_belief)
        for i in range(N_AGENTS):
            xs = [c[0] for c in traj_cells[i]]
            ys = [c[1] for c in traj_cells[i]]
            lines_b[i].set_data(xs, ys)
        fig_b.canvas.draw()
        buf_b = io.BytesIO()
        fig_b.savefig(buf_b, format='png', bbox_inches='tight')
        buf_b.seek(0)
        belief_writer.append_data(imageio.imread(buf_b))
        buf_b.close()

        # --- 업데이트 Figure 2: truth map with path ---
        for i in range(N_AGENTS):
            xs = [c[0] for c in traj_cells[i]]
            ys = [c[1] for c in traj_cells[i]]
            lines_t[i].set_data(xs, ys)
        fig_t.canvas.draw()
        buf_t = io.BytesIO()
        fig_t.savefig(buf_t, format='png', bbox_inches='tight')
        buf_t.seek(0)
        truth_writer.append_data(imageio.imread(buf_t))
        buf_t.close()

        # --- 업데이트 Figure 3: patches for each agent ---
        for i, a in enumerate(agents):
            axes_p[i].imshow(a.patch, cmap='gray', vmin=0, vmax=2)
            axes_p[i].set_title(f"A{i} patch", fontsize=8)

        fig_p.canvas.draw()
        buf_p = io.BytesIO()
        fig_p.savefig(buf_p, format='png', bbox_inches='tight')
        buf_p.seek(0)
        patch_writer.append_data(imageio.imread(buf_p))
        buf_p.close()

        step += 1

    # 종료 처리
    belief_writer.close()
    truth_writer.close()
    patch_writer.close()
    plt.close(fig_b)
    plt.close(fig_t)
    plt.close(fig_p)

    print("Generated: belief_update.gif, truth_path_heading.gif, patches.gif")

if __name__ == "__main__":
    main()
