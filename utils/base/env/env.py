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
        self.robot_2d_velocities = np.zeros((self.num_agent, 2), dtype=np.float32)
        self.robot_yaw_rate = np.zeros((self.num_agent, 1), dtype=np.float32)

        self.num_step = 0
        self.reached_goal = np.zeros((self.cfg.num_agent, 1), dtype=np.bool_)

        # Optional plotting setup
        if self.plot:
            self.frame_files = []

        # Optional plotting setup
        self.infos = {}


    def reset(self, episode_index: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if episode_index is not None:
            self.episode_index = episode_index
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
        self.angles = 0 * np.random.uniform(0, 2*np.pi, size=self.num_agent)
        # Perform initial sensing update for each agent
        for i in range(self.num_agent):
            cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            
            self.robot_belief = sensor_work_heading(
                cell,
                round(self.sensor_range / self.cell_size),
                self.robot_belief,
                self.ground_truth,
                np.rad2deg(self.angles[i]),
                360,
                self.map_mask
            )

        self.belief_info.map = self.robot_belief.copy()
        self.prev_dists = [np.linalg.norm(self.robot_locations[i] - self.goal_coords) for i in range(self.num_agent)]
        
        self._compute_intermediate_values()
        self.obs_buf = self._get_observations()
        self.state_buf = self._get_states()
        self._update_infos()

        return self.obs_buf, self.state_buf, self.infos


    def _set_goal_state(self) -> np.ndarray:
        goal_cells = np.column_stack(np.nonzero(self.ground_truth == self.map_mask["goal"]))
        num_samples = min(self.num_agent, len(goal_cells))

        goal_indices = np.random.choice(len(goal_cells), size=num_samples)
        sampled_cells = goal_cells[goal_indices]

        rows = sampled_cells[:, 0]
        cols = sampled_cells[:, 1]

        x_coords = self.belief_origin_x + cols * self.cell_size
        y_coords = self.belief_origin_y + (self.ground_truth_size[0] - 1 - rows) * self.cell_size

        sampled_goal_world = np.column_stack((x_coords, y_coords))

        return np.mean(sampled_goal_world, axis=0)

    def _set_init_state(self) -> Tuple[np.ndarray, np.ndarray]:
        H = self.ground_truth.shape[0]
        start_cells = np.column_stack((np.nonzero(self.ground_truth == self.map_mask["start"])[0], 
                                       np.nonzero(self.ground_truth == self.map_mask["start"])[1]))
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

        # 2) 그레이스케일 → 클래스 매핑 : belief map representation과 겹치는 부분은 index 통일
        #   230 → free (0)
        #    1  → unknown (1)
        #   128 → obstacle (2)
        #    80 → goal     (3)
        #   200 → start    (4)
        gray2class = {
            230: self.map_mask["free"],
            200: self.map_mask["start"],
            128: self.map_mask["occupied"],
            80 : self.map_mask["goal"]
        }

        # 기본값(예: unknown)은 1로 처리
        ground_truth = np.full_like(arr, fill_value=1, dtype=np.uint8)
        for gray, cls in gray2class.items():
            ground_truth[arr == gray] = cls

        # 3) 시작 셀 찾기
        ys, xs = np.nonzero(ground_truth == self.map_mask["start"])
        if len(ys) == 0:
            raise ValueError(f"No start zone (value=2) found in map {map_path}")
        robot_cell = np.array([ys[0], xs[0]])

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location) -> None:
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell, heading) -> None:
        # robot_belief와 belief_info 내부의 map을 업데이트 및 동기화
        self.robot_belief = sensor_work_heading(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth, heading, self.fov, self.map_mask)
        self.belief_info.map = self.robot_belief.copy()


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
        
        # Action 전처리 단계
        self._pre_apply_action(actions)
        for i, action in enumerate(self.actions):
            # 이미 도달한 에이전트는 상태 업데이트 X
            if self.reached_goal[i]:
                continue
            
            # ============== Step Numerical Simulation ================

            # action을 적용하여 robot state (위치 및 각도) 업데이트
            self._apply_action(i, action)

            # ========================================================

            # Belief 업데이트
            cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            self.update_robot_belief(cell, np.rad2deg(self.angles[i]))

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

        return self.obs_buf, self.state_buf, self.reward_buf, self.termination_buf, self.truncation_buf, self.infos
    

    # =============== Base Env Methods ===================

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
    def _pre_apply_action(self, actions):
        raise NotImplementedError(f"Please implement the '_pre_apply_action' method for {self.__class__.__name__}.") 

    @abstractmethod
    def _apply_action(self, agent_id: int, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"Please implement the '_apply_action' method for {self.__class__.__name__}.") 
    

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