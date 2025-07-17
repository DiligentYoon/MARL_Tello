import torch
import numpy as np
from utils.utils import *
from parameter import *

from torch.nn import Module
from .env import Env

class Agent:
    def __init__(self, 
                 id: int, 
                 policy_net: Module, 
                 env:Env, 
                 cfg: dict,
                 device: torch.device = torch.device("cuda")):
        
        self.cfg = cfg
        self.id          = id
        self.device      = device
        self.policy_net  = policy_net

        # 환경에서 주어진 시작 위치, 헤딩으로 초기 상태 설정
        init_loc = env.robot_locations[id]      # env.step 이전에 Env.__init__에서 세팅된 위치
        init_heading = env.angles[id]
        self.location = np.array(init_loc, dtype=np.float32)
        self.heading  = float(init_heading)
        self.travel_dist = 0.0

        # 센서/모션 파라미터
        self.fov          = FOV
        self.sensor_range = SENSOR_RANGE
        self.max_velocity = MAX_VELOCITY
        self.max_yaw_rate = MAX_YAW_RATE
        self.cell_size    = CELL_SIZE

        self.prev_velocity    = 0
        self.prev_yaw_rate    = 0

        # 환경과 맵 정보 연결
        self.env = env
        self.map_info = env.belief_info

        # 학습용 버퍼
        self.episode_buffer = [[] for _ in range(NUM_EPISODE_BUFFER)]

        # 로컬 맵 Patch
        self.patch = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=env.robot_belief.dtype)  

        # 시각화용 궤적 기록
        self.plot = self.cfg["plot"]
        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []


    def update_state(self, new_loc, new_heading):
        # travel distance 계산
        if self.location is not None:
            self.travel_dist += np.linalg.norm(new_loc - self.location)
        self.location = new_loc
        self.heading  = new_heading

        if self.plot:
            self.trajectory_x.append(new_loc[0])
            self.trajectory_y.append(new_loc[1])

    def get_observation(self, all_drones):
        """
        관측(observation)을 다음 형태의 1D 벡터로 반환합니다:
           [APF_x, APF_y,
            pos_x, pos_y, heading,
            vel_x, vel_y,
            nbr_pos_x, nbr_pos_y, nbr_vel_x, nbr_vel_y]
        """

        drone_cell = get_cell_position_from_coords(self.location, self.map_info)  # [col, row]
        belief_map = self.env.robot_belief                                     # 2D grid
        goal_coord = self.env.goal_coords                                      # [x, y] world
 
        # 1) APF 벡터 (shape: (2,))
        apf_vec, patch = self.compute_apf_patch(drone_cell, belief_map, goal_coord, self.map_info)
        apf_vec = np.asarray(apf_vec).ravel()    # 1D로 변환
        self.patch = patch

        # 2) 내 위치(x,y)와 heading
        pos = np.array([self.location[0],
                        self.location[1],
                        self.heading], dtype=np.float32)   # (3,)

        # 3) 내 선속도를 x,y 성분으로 분해
        #    self.prev_velocity 는 스칼라 속도 (m/s)
        rad = np.radians(self.heading)
        vel_x = self.prev_velocity * np.cos(rad)
        vel_y = self.prev_velocity * np.sin(rad)
        vel = np.array([vel_x, vel_y], dtype=np.float32)  # (2,)

        # 4) 가장 가까운 드론 찾기 (pos_x, pos_y, vel_x, vel_y)
        min_dist = float('inf')
        nbr_state = np.zeros(4, dtype=np.float32)  # [pos_x, pos_y, vel_x, vel_y]
        for other in all_drones:
            if other.id == self.id or other.location is None:
                continue
            d = np.linalg.norm(other.location - self.location)
            if d < min_dist:
                min_dist = d
                # 위치
                nbr_state[0] = other.location[0]
                nbr_state[1] = other.location[1]
                # 속도 (분해)
                oth_rad = np.radians(other.heading)
                nbr_state[2] = other.prev_velocity * np.cos(oth_rad)
                nbr_state[3] = other.prev_velocity * np.sin(oth_rad)

        # 5) 모두 하나의 1D 벡터로 concat
        obs = np.concatenate((apf_vec,pos,vel,nbr_state),axis=0)  # shape = (2 + 3 + 2 + 4) = 11
        obs = np.round(obs, 3)

        # 6) torch 텐서로 변환: (1,11)
        return torch.from_numpy(obs).float().unsqueeze(0).to(self.device)



    def select_action(self, observation, deterministic=False):
        with torch.no_grad():
            mean, log_std = self.policy_net(observation)
            std = log_std.exp()
        if deterministic:
            action = mean
        else:
            dist   = torch.distributions.Normal(mean, std)
            action = dist.rsample()

        # 클리핑
        v   = action[0,0].clamp(0, self.max_velocity)
        yaw = action[0,1].clamp(-self.max_yaw_rate, self.max_yaw_rate)

        # 다음 스텝용 보관
        self.prev_velocity = v.item()
        self.prev_yaw_rate = yaw.item()

        return v.item(), yaw.item()
    
    def record_transition(self):
        # To Do..
        pass

    def save_observation(self, obs):
        self.episode_buffer[0].append(obs)

    def save_action(self, action):
        a = torch.tensor(action, device=self.device).unsqueeze(0)
        self.episode_buffer[1].append(a)

    def save_reward(self, reward):
        r = torch.tensor([[[reward]]], device=self.device)
        self.episode_buffer[2].append(r)

    def save_next_observation(self, obs2):
        self.episode_buffer[3].append(obs2)

    def save_done(self, done):
        d = torch.tensor([[[int(done)]]], device=self.device)
        self.episode_buffer[4].append(d)


    def compute_apf_patch(
        self,
        drone_cell: np.ndarray,        # [col, row] in cell indices
        belief_map: np.ndarray,        # 2D grid of ints (0=free, 1=unknown, 2=obstacle, etc.)
        goal_coord: np.ndarray,        # [x, y] in world units (cells × cell_size)
        map_info: MapInfo,

        cell_size: float = 0.01,
        patch_size: int = 30,
        k_rep: float = 0.005,
        k_att: float = 1.0) -> (np.ndarray, np.ndarray):
        """
        Extract a patch around the drone and compute APF.

        Returns:
        - apf_vec: 2D APF vector [vx, vy]
        - patch:   the extracted patch (patch_size × patch_size)
        """
        H, W = belief_map.shape
        half = patch_size // 2
        
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
        patch = np.ones((patch_size, patch_size), dtype=belief_map.dtype)  # UNKNOWN=1
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
            dx = rel_col * cell_size
            dy = - rel_row * cell_size

            # continuous vector from obstacle to drone (i.e. repulsion direction)
            diff = np.array([-dx, -dy], dtype=float)

            dist2 = diff.dot(diff) + eps
            # repulsive force ∝ diff / dist²
            f_rep += k_rep * (diff / dist2)

        # 3) Attractive term: unit vector toward goal
        drone_pos = get_coords_from_cell_position(np.array([c, r]), map_info)
        diff = goal_coord - drone_pos
        norm = np.linalg.norm(diff) + eps
        f_att = k_att * (diff / norm)

        # print(f"f_rep: {f_rep}")
        # print(f"f_att: {f_att}")
        # 4) Sum and return both APF vector and the patch
        apf_vec = f_att + f_rep
        # print(f"f_att: {apf_vec}")
        return apf_vec, patch