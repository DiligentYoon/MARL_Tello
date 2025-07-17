import os
import numpy as np
from PIL import Image
from abc import abstractmethod
from typing import Tuple

from .env_cfg import EnvCfg
from utils.utils import *


class MapInfo:
    def __init__(self, map=None, map_origin_x=None, map_origin_y=None, cell_size=None, map_mask=None):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size
        self.map_mask = map_mask

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y



class Env():
    def __init__(self, episode_index: int | np.ndarray, cfg: EnvCfg) ->None:
        self.cfg = cfg
        self.episode_index = episode_index

        self.seed = self.cfg.seed
        self.dt = self.cfg.physics_dt
        self.plot = self.cfg.plot

        self.fov = self.cfg.fov
        self.sensor_range = self.cfg.sensor_range
        self.num_agent = self.cfg.num_agent
        self.max_lin_vel = self.cfg.max_velocity
        self.max_ang_vel = self.cfg.max_yaw_rate

        self.map_mask = self.cfg.map_representation
        self.cell_size = self.cfg.cell_size

        self.belief_info = MapInfo(map = None,
                                   map_origin_x=0.0,
                                   map_origin_y=0.0,
                                   cell_size=self.cell_size,
                                   map_mask=self.map_mask)
        
        # Location은 2D, Velocity는 스칼라 커맨드
        self.robot_locations = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.robot_velocities = np.zeros((self.num_agent, 1), dtype=np.float32)

        self.num_step = 0
        self.reached_goal = np.zeros((self.cfg.num_agent, 1), dtype=np.bool_)

        # Optional plotting setup
        if self.plot:
            self.frame_files = []

        # Optional plotting setup
        self.infos = {}


    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        # Load ground truth map and initial cell
        self.num_step = 0
        self.ground_truth, _ = self.import_ground_truth(self.episode_index)
        self.ground_truth_size = self.ground_truth.shape

        # Initialize belief map
        self.robot_belief = np.ones(self.ground_truth_size, dtype=np.uint8) * self.map_mask["unknown"]
        self.belief_origin_x = 0.0
        self.belief_origin_y = 0.0

        # Randomly place N_AGENTS in start zone (value == 2)
        world_x, world_y = self._set_init_state()
        self.robot_locations = np.stack([world_x, world_y], axis=1)

        # Compute goal coordinates (average over all goal cells)
        goal_world =self._set_goal_state()
        self.goal_coords = goal_world

        # Prepare belief_info for coordinate conversions
        self.belief_info.map = self.robot_belief
        self.belief_info.map_origin_x = self.belief_origin_x
        self.belief_info.map_origin_y = self.belief_origin_y

        # Initialize headings
        self.angles = np.random.uniform(0, 360, size=self.num_agent)
        # Perform initial sensing update for each agent
        for i in range(self.num_agent):
            cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            
            self.robot_belief = sensor_work_heading(
                cell,
                round(self.sensor_range / self.cell_size),
                self.robot_belief,
                self.ground_truth,
                self.angles[i],
                360,
                self.map_mask
            )

        self.prev_dists = [np.linalg.norm(self.robot_locations[i] - self.goal_coords) for i in range(self.num_agent)]
        
        self._compute_intermediate_values()
        self.obs_buf = self._get_observations()
        self._update_infos()

        return self.obs_buf, self.infos


    def _set_goal_state(self) -> np.ndarray:
        goal_cells = np.column_stack(np.nonzero(self.ground_truth == 3))
        goal_world = []

        for row, col in goal_cells:
            x = self.belief_origin_x + col * self.cell_size
            y = self.belief_origin_y + (self.ground_truth_size[0] - 1 - row) * self.cell_size
            goal_world.append([x, y])

        return np.mean(goal_world, axis=0)

    def _set_init_state(self) -> Tuple[np.ndarray, np.ndarray]:
        H = self.ground_truth.shape[0]
        start_cells = np.column_stack((np.nonzero(self.ground_truth == 2)[0], np.nonzero(self.ground_truth == 2)[1]))
        idx = np.random.choice(len(start_cells), self.num_agent, replace=False)
        chosen = start_cells[idx]
        rows, cols = chosen[:, 0], chosen[:, 1]
        world_x = self.belief_origin_x + cols * self.cell_size
        world_y = self.belief_origin_y + (H -1 - rows) * self.cell_size
        return world_x, world_y



    def import_ground_truth(self, episode_index) -> Tuple[np.ndarray, np.ndarray]:
        map_dir = os.path.join(os.getcwd(), "map")
        map_list = sorted(os.listdir(map_dir))
        map_index = episode_index % len(map_list)
        map_path = os.path.join(map_dir, map_list[map_index])

        # 1) PNG 로드
        img = Image.open(map_path).convert('L')
        arr = np.array(img, dtype=np.uint8)

        # 2) 그레이스케일 → 클래스 매핑
        #   230 → free (0)
        #   128 → obanglesstacle (1)
        #   200 → start    (2)
        #    80 → goal     (3)
        gray2class = {230: 0, 128: 1, 200: 2, 80: 3}

        # 기본값(예: unknown)은 1로 처리
        ground_truth = np.full_like(arr, fill_value=1, dtype=np.uint8)
        for gray, cls in gray2class.items():
            ground_truth[arr == gray] = cls

        # 3) 시작 셀 찾기
        ys, xs = np.nonzero(ground_truth == 2)
        if len(ys) == 0:
            raise ValueError(f"No start zone (value=2) found in map {map_path}")
        robot_cell = np.array([ys[0], xs[0]])

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location) -> None:
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell, heading) -> None:
        self.robot_belief = sensor_work_heading(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth, heading, self.fov, self.map_mask)


    def step(self, actions) -> Tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     dict[str, np.ndarray]]:
        """
            actions (n x 2)
                [n, 0] : linear velocity command of n'th agent
                [n, 1] : angular velocity command of n'th agent

            Return :
                obs_buf -> [n, obs_dim]         : t+1 observation
                state_buf -> [n, state_dim]     : t+1 state
                action_buf -> [n, act_dim]      : t action
                reward_buf -> [n, 1]            : t+1 reward
                termination_buf -> [n, 1]       : t+1 terminated
                truncation_buf  -> [n, 1]       : t+1 truncated
                info -> dict[str, [n, dim]]     : additional metric 

        """
        
        # stochastic action인 경우를 대비, 한번 더 클램핑
        self._pre_apply_action(actions)

        # apply action : 연구에서는 고정할 것이기 때문에 따로 함수로 안뺌.
        for i, (v, yaw_rate) in enumerate(self.actions):
            # 이미 도달한 에이전트는 상태 업데이트 X
            if self.reached_goal[i]:
                continue
            
            # ============== Step Numerical Simulation ================

            # 값 계산
            angle_new = (self.angles[i] + yaw_rate * self.dt) % 360
            dx = v * self.dt * np.cos(np.radians(angle_new))
            dy = v * self.dt * np.sin(np.radians(angle_new))

            # 위치 및 각도 업데이트
            self.angles[i] = angle_new
            self.robot_locations[i] += np.array([dx, dy])

            # Belief 업데이트
            cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            self.update_robot_belief(cell, angle_new)
        
        # Potential Based Reward Shaping을 수행하기 위한 현재 거리 계산
        self.cur_dists = [np.linalg.norm(self.robot_locations[i] - self.goal_coords) for i in range(self.num_agent)]

        # Done 신호 생성
        self.num_step += 1
        self.termination_buf, self.truncation_buf, self.reached_goal = self._get_dones()

        # 보상 계산
        self.reward_buf = self._get_rewards()
        
        # Next Observation 세팅
        self.obs_buf = self._get_observations()
        self.state_buf = self._get_states()

         # ======== 추가 정보 infos 업데이트 ===========
        self._update_infos()
        self.prev_dists = self.cur_dists

        return self.obs_buf, self.state_buf, self.reward_buf, self.termination_buf, self.truncation_buf, self.infos
    

    # =============== Base Env Methods ===================

    def _pre_apply_action(self, actions):
        """
            actions (n x 2)
                [n, 0] : linear velocity command of n'th agent
                [n, 1] : angular velocity command of n'th agent

        """
        self.actions = actions
        self.actions[:, 0] = np.clip(actions[:, 0] * self.max_lin_vel,
                                     -self.max_lin_vel,
                                      self.max_lin_vel)
        
        self.actions[:, 1] = np.clip(actions[:, 1],
                                     -self.max_ang_vel,
                                      self.max_ang_vel)
        
        self.robot_velocities[:, :] = self.actions[:, 0].reshape(-1, 1)



    def evaluate_exploration_rate(self):
        """
        Updates self.explored_rate to be the fraction of map cells
        that the agents have observed (i.e., not UNKNOWN).
        """
        # Assume UNKNOWN is the constant for unseen cells, imported from parameter
        # robot_belief is a 2D numpy array of values {FREE, OCCUPIED, UNKNOWN}
        belief = self.robot_belief
        total_cells = belief.size
        # Count cells that are not UNKNOWN
        explored_cells = np.count_nonzero(belief != self.map_mask['unknown'])
        # Compute exploration rate
        return explored_cells / total_cells
    


    def _update_infos(self):
        self.infos["explored_rate"] = self.evaluate_exploration_rate()



    # =============== Env-Specific Abstract Methods =================

    @abstractmethod
    def _get_observations(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_observations' method for {self.__class__.__name__}.")
    

    @abstractmethod
    def _get_states(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_states' method for {self.__class__.__name__}.")
    

    @abstractmethod
    def _get_dones(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_dones' method for {self.__class__.__name__}.")


    @abstractmethod
    def _compute_intermediate_values(self) -> None:
        raise NotImplementedError(f"Please implement the '_compute_intermediate_values' method for {self.__class__.__name__}.")


    @abstractmethod
    def _get_rewards(self) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_get_rewards' method for {self.__class__.__name__}.")

    
    # ===============================================================


    # # ================ 삭제 후보 ==================== # 


    # # 해당 함수는 specific env 코드에 넣고, intermediate 값 계산과 함께 사용
    # def check_done(self) -> bool:
    #     """
    #     Returns:
    #         0  = 계속 진행
    #         1  = 목표 도달
    #         -1  = 장애물 충돌 또는 에이전트 간 충돌
    #     """
    #     # 1) 로봇 셀 좌표
    #     cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
    #     rows, cols = cells[:, 0], cells[:, 1]
        
    #     # 2) 종료 조건 판정
    #     in_goal      = np.all(self.ground_truth[rows, cols] == 3)
    #     hit_obstacle = np.any(self.ground_truth[rows, cols] == 1)
    #     # 에이전트 간 충돌 (동일 셀 공유)
    #     flat_idx  = rows * self.ground_truth.shape[1] + cols
    #     counts    = np.bincount(flat_idx)
    #     collision = np.any(counts > 1)

    #     # 3) 상태 코드 결정
    #     if in_goal:
    #         code = 1
    #     elif hit_obstacle or collision:
    #         code = -1
    #     else:
    #         code = 0

    #     # 4) self.done 설정
    #     self.done = (code != 0)

    #     # 5) 이유 출력
    #     if code == 1:
    #         print("[Done] 종료 사유: 목표 도달")
    #     elif code == -1:
    #         reason_list = []
    #         if hit_obstacle:
    #             reason_list.append("장애물 충돌")
    #         if collision:
    #             reason_list.append("에이전트 간 충돌")
    #         print(f"[Done] 종료 사유: {', '.join(reason_list)}")

    #     return code




    # def heading_to_vector(self, heading, length=1):
    #     # Convert heading to vector
    #     if isinstance(heading, (list, np.ndarray)):
    #         heading = heading[0]
    #     heading_rad = np.radians(heading)
    #     return np.cos(heading_rad) * length, np.sin(heading_rad) * length


    # def create_sensing_mask(self, location, heading):
    #     mask = np.zeros_like(self.ground_truth)

    #     location_cell = get_cell_position_from_coords(location, self.belief_info)
    #     # Create a Point for the robot's location
    #     robot_point = Point(location_cell)
    #     # heading = heading*(360/self.num_angles)

    #     # Calculate the angles for the sector
    #     start_angle = (heading - self.fov / 2 + 360) % 360
    #     end_angle = (heading + self.fov / 2) % 360

    #     # Create points for the sector
    #     sector_points = [robot_point]
    #     if start_angle <= end_angle:
    #         angle_range = np.linspace(start_angle, end_angle, 20)
    #     else:
    #         angle_range = np.concatenate([np.linspace(start_angle, 360, 10), np.linspace(0, end_angle, 10)])
    #     for angle in angle_range: 
    #         x = location_cell[0] + self.sensor_range/self.cell_size * np.cos(np.radians(angle))
    #         y = location_cell[1] + self.sensor_range/self.cell_size * np.sin(np.radians(angle))
    #         sector_points.append(Point(x, y))
    #     sector_points.append(robot_point) 

    #     sector = Polygon(sector_points)

    #     x_coords, y_coords = sector.exterior.xy
    #     y_coords = np.rint(y_coords).astype(int)
    #     x_coords = np.rint(x_coords).astype(int)
    #     rr, cc = sk_polygon(
    #             [int(round(y)) for y in y_coords],
    #             [int(round(x)) for x in x_coords],
    #             shape=mask.shape
    #         )
        
    #     free_connected_map = get_free_and_connected_map(location, self.belief_info)

    #     mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])
       
    #     return mask
    
    # ============================================== # 