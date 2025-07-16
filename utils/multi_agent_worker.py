"""
A multi-agent worker class for coordinating multi-robots exploration in an indoor environment.

This class manages a group of agents performing collaborative exploration, handling 
their movement, observation, reward calculation, and simulation steps. It supports 
features like collision avoidance, trajectory planning, and performance tracking.

Key functionalities:
- Initializes multiple agents with a shared policy network
- Runs exploration episodes with collision resolution
- Tracks agent locations, headings, and exploration progress
- Generates visualizations of the exploration process
- Calculates rewards and saves episode data

Attributes:
    meta_agent_id (int): Identifier for the meta-agent group
    global_step (int): Current global simulation step
    env (Env): Environment simulation instance
    robot_list (List[Agent]): List of agents in the exploration team
    episode_buffer (List): Buffer for storing episode data
    perf_metrics (dict): Performance metrics for the episode
"""
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Wedge, FancyArrowPatch

from utils.env import Env
from utils.agent import Agent
from utils.utils import *
from utils.model import MLPPolicy


if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class MultiAgentWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.fov = FOV
        self.sensor_range = SENSOR_RANGE
        self.sim_steps = NUM_SIM_STEPS

        self.env = Env(global_step, self.fov, self.sensor_range, plot=self.save_image)
        self.n_agents = N_AGENTS

        self.robot_list = [Agent(i, policy_net, self.env, self.fov, self.env.angles[i], self.sensor_range, self.device, self.save_image)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(NUM_EPISODE_BUFFER):
            self.episode_buffer.append([])


    def run_episode(self):
        done = False

        # 에피소드 루프
        for step in range(MAX_EPISODE_STEP):
            # 1) 관측 : env로 교체
            observations = [
                agent.get_observation(self.robot_list)
                for agent in self.robot_list
            ]
            # 2) 행동 : agent 유지
            actions = [
                agent.select_action(obs)
                for agent, obs in zip(self.robot_list, observations)
            ]
            # 3) 환경 스텝 : env로 교체
            rewards, done, info = self.env.step(actions)

            for i, agent  in enumerate(self.robot_list):
                new_loc     = self.env.robot_locations[i]
                new_heading = self.env.angles[i]
                agent.update_state(new_loc, new_heading)
            
            # 4) next 관측 -> 삭제 (step에서 가져옴)
            next_observations = [
                agent.get_observation(self.robot_list)
                for agent in self.robot_list
            ]

            # 4) 버퍼 저장
            for i, agent in enumerate(self.robot_list):
                agent.save_observation(observations[i])
                agent.save_action(actions[i])
                agent.save_reward(rewards[i])
                agent.save_next_observation(next_observations[i])
                agent.save_done(done)

            if done != 0:
                break

        # 5) 퍼포먼스 메트릭
        self.perf_metrics['travel_dist']  = max(a.travel_dist for a in self.robot_list)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = int(done) #1: sucess, -1: collision

        # 6) 에피소드 전체 버퍼 합치기
        for agent in self.robot_list:
            for idx in range(NUM_EPISODE_BUFFER):
                self.episode_buffer[idx] += agent.episode_buffer[idx]


if __name__ == '__main__':
    from parameter import *
    import torch
    policy_net = MLPPolicy(NODE_INPUT_DIM, NUM_ANGLES_BIN)
    if LOAD_MODEL:
        checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy'])
        print('Policy loaded!')
    worker = MultiAgentWorker(0, policy_net, 888, 'cpu', True)
    worker.run_episode()
