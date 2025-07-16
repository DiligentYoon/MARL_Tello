import numpy as np

from typing import Tuple
from utils.sensor import *
from utils.utils import *
from .env import Env, MapInfo

from .env_apf_cfg import APFEnvCfg



class APFEnv(Env):
    cfg: APFEnvCfg
    def __init__(self, episode_index: int | np.ndarray, cfg: APFEnvCfg):
        super().__init__(episode_index, cfg)

        self.dt = self.cfg.physics_dt
        self.decimation = self.cfg.decimation
        self.max_episode_steps = self.cfg.max_episode_steps

        self.patch_size = self.cfg.patch_size


        self.local_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        self.APF_vec = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.neighbor_states = np.zeros((self.num_agent, 4), dtype=np.float32)

        # Done flags for reward shaping
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)


        # Optional Additional State
        self.infos["explored_rate"] = np.zeros((1, ), dtype=np.float32)

    def reset(self):
        # 나머지 플래그는 사용하기 전 계산 되므로 초기화 X
        self.is_first_reached = np.ones((self.num_agent, 1), dtype=np.bool_)
        return super().reset()


    def _compute_intermediate_values(self):
        """
            업데이트된 state값들을 바탕으로, obs값에 들어가는 planning state 계산
        """
        # APF 벡터 계산
        for i in range(self.num_agent):
            drone_cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            apf_vec, _ = self.compute_apf_patch(drone_cell, self.robot_belief, self.goal_coords, self.belief_info)
            self.APF_vec[i] = apf_vec

        # 가장 가까운 이웃 드론 상태 계산
        for i in range(self.num_agent):
            min_dist_sq = float('inf')
            closest_neighbor_state = np.zeros(4, dtype=np.float32)
            for j in range(self.num_agent):
                if i == j:
                    continue
                
                dist_sq = np.sum(np.square(self.robot_locations[i] - self.robot_locations[j]))
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    # 이웃 위치
                    closest_neighbor_state[0] = self.robot_locations[j, 0]
                    closest_neighbor_state[1] = self.robot_locations[j, 1]
                    # 이웃 속도
                    rad = np.radians(self.angles[j])
                    vx = self.robot_velocities[j] * np.cos(rad)
                    vy = self.robot_velocities[j] * np.sin(rad)
                    closest_neighbor_state[2] = vx
                    closest_neighbor_state[3] = vy
            
            self.neighbor_states[i] = closest_neighbor_state


    
    def _get_observations(self) -> np.ndarray:
        """
            Observation Config for Actor Network [n, 11]
                1. APF_vector : [n, 2]
                2. Position (x, y, yaw) : [n , 3]
                3. Velocity (vx, vy) : [n, 2]
                4. closest 2D Location : [n, 2]
                5. closest 2D Velocity : [n, 2]
        """
        # APF_vector [n, 2]
        apf_vectors = self.APF_vec

        # 자신의 Position (x, y, yaw) [n, 3]
        positions = np.hstack((self.robot_locations, self.angles[:, np.newaxis]))

        # 자신의 Velocity (vx, vy) [n, 2]
        rads = np.radians(self.angles)
        vel_x = self.robot_velocities * np.cos(rads).reshape(-1, 1)
        vel_y = self.robot_velocities * np.sin(rads).reshape(-1, 1)
        velocities = np.hstack((vel_x, vel_y))

        # Closest neighbor state [n, 4] (pos_x, pos_y, vel_x, vel_y)
        neighbor_states = self.neighbor_states

        # Concatenate
        obs = np.hstack((apf_vectors, positions, velocities, neighbor_states))
        
        return obs
    

    def _get_states(self) -> np.ndarray:
        """
            State Config for Critic Network [n, 13]
                1. APF_vector : [n, 2]
                2. Position (x, y, yaw) : [n , 3]
                3. Velocity (vx, vy) : [n, 2]
                4. closest 2D Location : [n, 2]
                5. closest 2D Velocity : [n, 2]
                6. Action : [n, act_dim]
        """
        # APF_vector [n, 2]
        apf_vectors = self.APF_vec

        # 자신의 Position (x, y, yaw) [n, 3]
        positions = np.hstack((self.robot_locations, self.angles[:, np.newaxis]))

        # 자신의 Velocity (vx, vy) [n, 2]
        rads = np.radians(self.angles)
        vel_x = self.robot_velocities * np.cos(rads).reshape(-1, 1)
        vel_y = self.robot_velocities * np.sin(rads).reshape(-1, 1)
        velocities = np.hstack((vel_x, vel_y))

        # Closest neighbor state [n, 4] (pos_x, pos_y, vel_x, vel_y)
        neighbor_states = self.neighbor_states

        # 자신의 action [n, 2]
        actions = self.actions

        state = np.hstack((apf_vectors, positions, velocities, neighbor_states, actions))

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

        # 드론 간 충돌
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
        reward =  np.array(self.prev_dists) - np.array(self.cur_dists)

        for i in range(self.num_agent):
            
            if self.is_reached_goal[i] and self.is_first_reached[i]:
                reward += self.cfg.reward_info["goal"]
                self.is_first_reached[i] = False
            
            if self.is_collided_drone[i] or self.is_collided_obstacle[i]:
                reward += self.cfg.reward_info["collision"]

        return reward



    # ============= Auxilary Methods ==============
    def compute_apf_patch(
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

        # 2) Repulsive term: 8-directional unit repulsion, magnitude ∝ 1/dist²
        f_rep = np.zeros(2, dtype=float)
        obs_idxs = np.argwhere(patch == 2)  # obstacle locations in patch coords
        eps = 1e-6

        for i, j in np.argwhere(patch == 2):
            rel_row = i - half
            rel_col = j - half
            dx = rel_col * self.cell_size
            dy = - rel_row * self.cell_size

            # continuous vector from obstacle to drone (i.e. repulsion direction)
            diff = np.array([-dx, -dy], dtype=float)

            dist2 = diff.dot(diff) + eps
            # repulsive force ∝ diff / dist²
            f_rep += self.cfg.apf_k_rep * (diff / dist2)

        # 3) Attractive term: unit vector toward goal
        drone_pos = get_coords_from_cell_position(np.array([c, r]), map_info)
        diff = goal_coord - drone_pos
        norm = np.linalg.norm(diff) + eps
        f_att = self.cfg.apf_k_att * (diff / norm)

        # print(f"f_rep: {f_rep}")
        # print(f"f_att: {f_att}")
        # 4) Sum and return both APF vector and the patch
        apf_vec = f_att + f_rep
        # print(f"f_att: {apf_vec}")

        return apf_vec, patch