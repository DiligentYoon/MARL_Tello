import numpy as np

from .apf_act_env_cfg import APFActEnvCfg

from typing import Tuple
from utils.utils import *
from utils.base.env.env import Env, MapInfo



class APFActEnv(Env):
    cfg: APFActEnvCfg
    def __init__(self, episode_index: int | np.ndarray, cfg: dict):
        cfg = APFActEnvCfg(cfg)
        super().__init__(episode_index, cfg)

        self.dt = self.cfg.physics_dt
        self.decimation = self.cfg.decimation
        self.max_episode_steps = self.cfg.max_episode_steps

        self.patch_size = self.cfg.patch_size

        # 핵심 Planning State
        self.robot_target_angle = np.zeros((self.num_agent, 1), dtype=np.float32)
        self.local_patches = np.zeros((self.num_agent, self.patch_size, self.patch_size), dtype=np.float32)
        self.closest_obstacle_states = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.closest_neighbor_states = np.zeros((self.num_agent, 4), dtype=np.float32)
        self.neighbor_states = np.zeros((self.num_agent, self.num_agent-1, 4), dtype=np.float32)

        # Done flags for reward shaping
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)


        # Optional Additional State
        self.infos["explored_rate"] = np.zeros((1, ), dtype=np.float32)

    def reset(self, episode_index: int = None):
        # 나머지 플래그는 사용하기 전 계산 되므로 초기화 X
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)
        return super().reset(episode_index)
    

    def _pre_apply_action(self, actions: np.ndarray) -> np.ndarray:
        """
            APF 기반의 속도벡터 생성
            
            Input :
                Actions (n, 3) : (Attractive_gain, Repulsive_gain, Inter-Drone_gain)
            Process :
                1. Post-Processing : [-1, 1]범위의 action값을 대응되는 range게 맞게 매핑
                2. APF 지배 방정식에 의한 Velocity 계산
                3. Max velocity값들로 클램핑
        """
        self.actions = actions.copy()

        self.actions[:, 0] = 0.5 * (self.actions[:, 0] + 1) * (self.cfg.apf_k_att_range[1] - self.cfg.apf_k_att_range[0]) + self.cfg.apf_k_att_range[0]
        self.actions[:, 1] = 0.5 * (self.actions[:, 1] + 1) * (self.cfg.apf_k_rep_range[1] - self.cfg.apf_k_rep_range[0]) + self.cfg.apf_k_rep_range[0]
        self.actions[:, 2] = 0.5 * (self.actions[:, 2] + 1) * (self.cfg.apf_k_inter_rep_range[1] - self.cfg.apf_k_inter_rep_range[0]) + self.cfg.apf_k_inter_rep_range[0]

        for i in range(self.num_agent):
            self.robot_2d_velocities[i, :] = self.compute_apf_vector(i)

        self.robot_2d_velocities[:, 0] = np.clip(self.robot_2d_velocities[:, 0],
                                                 -self.max_lin_vel,
                                                  self.max_lin_vel)
        
        self.robot_2d_velocities[:, 1] = np.clip(self.robot_2d_velocities[:, 1],
                                                 -self.max_lin_vel,
                                                  self.max_lin_vel)
        
        self.robot_velocities[:] = np.linalg.norm(self.robot_2d_velocities, axis=1).reshape(-1, 1)
        
        self.robot_target_angle = np.atan2(self.robot_2d_velocities[:, 1], self.robot_2d_velocities[:, 0])

        # print(f"vel : {self.robot_2d_velocities}")
        
    def _apply_action(self, agent_id, action):
        deg_error = self.robot_target_angle[agent_id] - self.angles[agent_id]

        if deg_error > np.pi:
            deg_error -= 2 * np.pi
        elif deg_error < -np.pi:
            deg_error += 2 * np.pi
        
        self.robot_yaw_rate[agent_id] = self.cfg.kp * deg_error
        new_angle = self.angles[agent_id] + self.robot_yaw_rate[agent_id] * self.dt
        self.angles[agent_id] = np.atan2(np.sin(new_angle), np.cos(new_angle))

        vx = self.robot_velocities[agent_id] * np.cos(self.angles[agent_id])
        vy = self.robot_velocities[agent_id] * np.sin(self.angles[agent_id])

        dx = vx * self.dt
        dy = vy * self.dt   
        self.robot_locations[agent_id] += np.array([dx, dy]).reshape(-1)
        


    def _compute_intermediate_values(self):
        """
            업데이트된 state값들을 바탕으로, obs값에 들어가는 planning state 계산
        """
        # Potential Based Reward Shaping을 수행하기 위한 현재 거리 계산
        self.cur_dists = [np.linalg.norm(self.robot_locations[i] - self.goal_coords) for i in range(self.num_agent)]

        # Belief맵 기반의 Local Patch 계산
        for i in range(self.num_agent):
            drone_cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            local_patch, obstacle = self.compute_local_patch_and_obstacle(drone_cell, self.robot_belief, self.goal_coords, self.belief_info)
            self.local_patches[i] = local_patch
            self.closest_obstacle_states[i] = obstacle

        # 가장 가까운 이웃 드론 상태 계산
        for i in range(self.num_agent):
            m = 0
            min_dist_sq = float('inf')
            closest_neighbor_state = np.zeros(4, dtype=np.float32)
            neighbor_state = np.zeros(4, dtype=np.float32)
            for j in range(self.num_agent):
                if i == j:
                    continue
                
                # 이웃 위치
                neighbor_state[0] = self.robot_locations[j, 0]
                neighbor_state[1] = self.robot_locations[j, 1]
                # 이웃 속도
                rad = self.angles[j]
                vx = self.robot_velocities[j] * np.cos(rad)
                vy = self.robot_velocities[j] * np.sin(rad)
                neighbor_state[2] = vx
                neighbor_state[3] = vy
                # 이웃 간 거리                
                dist_sq = np.sum(np.square(self.robot_locations[i] - self.robot_locations[j]))
                # 이웃 상태 저장 (N, N-1, 4)
                self.neighbor_states[i][m] = neighbor_state
                m += 1

                # 최소 이웃 상태 저장 로직
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_neighbor_state = neighbor_state
            # 최소 이웃 상태 저장 (N, 4)
            self.closest_neighbor_states[i] = closest_neighbor_state


    
    def _get_observations(self) -> np.ndarray:
        """
            Observation Config for Actor Network [n, obs_dim]
                1. Position (gx, gy, yaw) : [n , 3]
                2. Velocity (vx, vy) : [n, 2]
                3. closest 2D Location : [n, 2]
                4. closest 2D Velocity : [n, 2]
                5. closest obstacle state : [n, 2]
        """
        # 자신의 Position (x, y, yaw) [n, 3]
        goal_coords = self.goal_coords.repeat((self.num_agent)).reshape(-1, 2)
        goal_info = goal_coords - self.robot_locations
        rads = self.angles.reshape(-1, 1)
        positions = np.hstack((goal_info, rads))

        # 자신의 Velocity (vx, vy) [n, 2]
        vel_x = self.robot_velocities * np.cos(rads)
        vel_y = self.robot_velocities * np.sin(rads)
        velocities = np.hstack((vel_x, vel_y))

        # Closest neighbor state [n, 4] (pos_x, pos_y, vel_x, vel_y)
        neighbor_states = self.closest_neighbor_states

        # Closest Obstacle state [n, 2] (pos_x, pos_y)
        obstacle_states = self.closest_obstacle_states

        # Concatenate
        obs = np.hstack((positions, velocities, neighbor_states, obstacle_states)).astype(np.float32)
        
        return obs
    

    def _get_states(self) -> np.ndarray:
        """
            State Config for Critic Network [n, state_dim]
                1. Position (x, y, yaw) : [n , 3]
                2. Velocity (vx, vy) : [n, 2]
                3. closest 2D Location : [n, 2]
                4. closest 2D Velocity : [n, 2]
        """
        # 자신의 Position (x, y, yaw) [n, 3]
        goal_coords = self.goal_coords.repeat((self.num_agent)).reshape(-1, 2)
        goal_info = goal_coords - self.robot_locations
        rads = self.angles.reshape(-1, 1)
        positions = np.hstack((goal_info, rads))

        # 자신의 Velocity (vx, vy) [n, 2]
        vel_x = self.robot_velocities * np.cos(rads)
        vel_y = self.robot_velocities * np.sin(rads)
        velocities = np.hstack((vel_x, vel_y))

        # Closest neighbor state [n, 4] (pos_x, pos_y, vel_x, vel_y)
        neighbor_states = self.closest_neighbor_states

        # Closest Obstacle state [n, 2] (pos_x, pos_y)
        obstacle_states = self.closest_obstacle_states

        state = np.hstack((positions, velocities, neighbor_states, obstacle_states)).astype(np.float32)

        return state


    def _get_dones(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            특정 종료조건 및 타임아웃 계산
            Return :
                1. terminated : 
                    1-1. 벽에 충돌
                    1-2. 드론끼리 충돌
                    1-3. 골 지점 도달 (이 땐 골 지점에 머물도록 강제)
                2. truncated :
                    2-1. 타임아웃

        """
        # Planning State 업데이트
        self._compute_intermediate_values()

        # ============== Done 계산 로직 ===================

        # ---- Truncated 계산 -----
        timeout = self.num_step >= self.max_episode_steps - 1
        truncated = np.full((self.num_agent, 1), timeout, dtype=np.bool_)


        # ---- Terminated 계산 ----
        terminated = np.zeros((self.num_agent, 1), dtype=np.bool_)

        # 로봇 셀 좌표 변환
        cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        rows, cols = cells[:, 1], cells[:, 0]

        # 목표 도달 유무 체크
        self.is_reached_goal = self.ground_truth[rows, cols] == self.map_mask["goal"]

        # 맵 경계 체크
        H, W = self.ground_truth.shape
        out_of_bounds = (rows < 0) | (rows >= H) | (cols < 0) | (cols >= W)

        # 유효한 셀에 대해서만 ground_truth 값 확인
        valid_indices = ~out_of_bounds
        valid_rows, valid_cols = rows[valid_indices], cols[valid_indices]

        # 장애물 충돌 (맵 밖 포함)
        hit_obstacle = np.zeros_like(out_of_bounds, dtype=np.bool_)
        hit_obstacle[valid_indices] = self.ground_truth[valid_rows, valid_cols] == self.map_mask["occupied"]
        self.is_collided_obstacle = (hit_obstacle | out_of_bounds)[:, np.newaxis]

        # 드론 간 충돌 (점유 셀이 겹치면 충돌 판단)
        flat_indices = rows * W + cols
        unique_indices, counts = np.unique(flat_indices, return_counts=True)
        collided_indices = unique_indices[counts > 1]
        
        self.is_collided_drone.fill(False)
        for idx in collided_indices:
            colliding_agents = np.where(flat_indices == idx)[0]
            for agent_idx in colliding_agents:
                self.is_collided_drone[agent_idx] = True

        # 목표 지점 도달
        all_reached_goal = np.all(self.reached_goal)

        # 최종 Terminated 조건
        # 개별 드론이 충돌하거나 모든 드론이 목표에 도달하면 종료
        # step로직에서 사용되는 개별 드론의 도착 유무도 함께 리턴
        terminated = self.is_collided_obstacle | self.is_collided_drone | all_reached_goal

        return terminated, truncated, self.is_reached_goal
    

    def _get_rewards(self):
        reward =  10 * (np.array(self.prev_dists) - np.array(self.cur_dists)).astype(np.float32)

        for i in range(self.num_agent):
            
            # Goal 도달 리워드
            if self.is_reached_goal[i] and self.is_first_reached[i]:
                reward[i] += self.cfg.reward_info["goal"]
                self.is_first_reached[i] = False
            
            # 충돌 페널티
            if self.is_collided_drone[i] or self.is_collided_obstacle[i]:
                reward[i] += self.cfg.reward_info["collision"]

        self.prev_dists = self.cur_dists

        # time penalty
        reward -= 1

        return reward



    # ============= Auxilary Methods ==============
    def compute_local_patch_and_obstacle(
        self,
        drone_cell: np.ndarray,        # [col, row] in cell indices
        belief_map: np.ndarray,        # 2D grid of ints (0=free, 1=unknown, 2=obstacle, etc.)
        goal_coord: np.ndarray,        # [x, y] in world units (cells × cell_size)
        map_info: MapInfo) -> Tuple[np.ndarray, np.ndarray]:
        """
            Extract a patch around the drone and compute APF.

            Returns:
            - apf_vec: 2D APF vector [vx, vy]
            - patch:   the extracted patch (patch_size x patch_size)
        """
        H, W = self.belief_info.map.shape
        half = self.patch_size // 2
        
        c, r = int(drone_cell[0]), int(drone_cell[1])
        
        # 1) 패치 범위: [r-half, r+half), [c-half, c+half)
        r0 = r - half
        r1 = r + half
        c0 = c - half
        c1 = c + half

        # 2) 맵 바깥 클램핑
        r0_clip, r1_clip = max(r0, 0), min(r1, H)
        c0_clip, c1_clip = max(c0, 0), min(c1, W)

        # 3) 패치 내 복사 시작 인덱스
        pr0 = r0_clip - r0    # if r0<0, pr0>0
        pc0 = c0_clip - c0

        # 4) 추출 & 복사
        patch = np.ones((self.patch_size, self.patch_size), dtype=belief_map.dtype)  # UNKNOWN=1
        h_src = r1_clip - r0_clip  # always <= patch_size
        w_src = c1_clip - c0_clip
        patch[pr0:pr0 + h_src, pc0:pc0 + w_src] = belief_map[r0_clip:r1_clip, c0_clip:c1_clip]

        # 5) 장애물 까지의 거리 계산
        obs_idxs = np.argwhere(patch == self.cfg.map_representation["occupied"])
        if obs_idxs.shape[0] == 0:
            # 장애물 없는경우
            return patch, np.array([half, half], dtype=np.float32)

        patch_center_cell = np.array([half, half])

        # 각 장애물들의 중심으로부터의 상대적 위치를 계산 (단위: 셀)
        # (obs_cell, 2) size
        relative_pos_cells = obs_idxs - patch_center_cell

        #  [cell_row, cell_col] -> [dx, dy] 변환
        relative_pos_world = relative_pos_cells[:, ::-1] * self.cell_size

        #    각 장애물까지의 유클리드 거리를 한 번에 계산합니다.
        distances = np.linalg.norm(relative_pos_world, axis=1)

        # 4. 계산된 거리들 중 가장 작은 값을 찾습니다.
        min_idx = np.argmin(distances)
        min_obs_state = relative_pos_world[min_idx, :]

        return patch, min_obs_state
    

    def compute_apf_vector(self, agent_id: int, eps=1e-6) -> np.ndarray:
        """
            Return:
                (2,) 크기의 Velocity Command를 APF로 계산
            
            Requurements:
                1. Repulsive Term (local patch 기준으로) 
                2. Inter-Drone Repulsive Term
                3. Goal Attractive Term
        
        """
        f_rep = np.zeros(2, dtype=np.float32)
        f_inter_rep = np.zeros(2, dtype=np.float32)
        f_att = np.zeros(2, dtype=np.float32)

        action = self.actions[agent_id, :]
        patch = self.local_patches[agent_id]

        # Attractive term
        drone_cell = get_cell_position_from_coords(self.robot_locations[agent_id], self.belief_info)
        c, r = int(drone_cell[0]), int(drone_cell[1])

        drone_pos = get_coords_from_cell_position(np.array([c, r]), self.belief_info).reshape(-1)
        diff = self.goal_coords - drone_pos
        norm = np.linalg.norm(diff) + eps
        f_att[:] = action[0] * diff

        # Obstacle repulsive term
        for i, j in np.argwhere(patch == self.cfg.map_representation["occupied"]):
            rel_row = i - self.cfg.patch_size/2
            rel_col = j - self.cfg.patch_size/2
            dx = rel_col * self.cell_size
            dy = - rel_row * self.cell_size

            diff = np.array([-dx, -dy], dtype=np.float32)
            dist = np.linalg.norm(diff) + eps

            if dist <= self.cfg.rho:
                f_rep += action[1] * (1/dist - 1/self.cfg.rho) * diff

        # Inter-Drone repulsive term
        neighbor_state = self.neighbor_states[agent_id] # (N-1, 4)
        for state in neighbor_state:
            diff = drone_pos - state[:2]
            dist = np.linalg.norm(diff) + eps

            if dist <= self.cfg.rho_agent:
                f_inter_rep +=  action[2] * (1/dist - 1/self.cfg.rho_agent) * diff

        return f_att + f_rep + f_inter_rep
