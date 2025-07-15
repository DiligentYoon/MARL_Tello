import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon
from PIL import Image


from utils.sensor import sensor_work_heading
from parameter import *
from utils.utils import *


class Env:
    def __init__(self, episode_index, fov, sensor_range, plot=False):
        self.episode_index = episode_index
        self.plot = plot

        # 1) Load ground truth map and initial cell
        self.ground_truth, _ = self.import_ground_truth(episode_index)
        H, W = self.ground_truth.shape
        self.ground_truth_size = (H, W)
        self.cell_size = CELL_SIZE  # meter

        # 2) Initialize belief map
        self.robot_belief = np.ones((H, W), dtype=np.uint8) * UNKNOWN
        self.belief_origin_x = 0.0
        self.belief_origin_y = 0.0
        self.sensor_range = sensor_range
        self.fov = fov

        
        # 3) Randomly place N_AGENTS in start zone (value == 2)
        H = self.ground_truth.shape[0]
        start_cells = np.column_stack((np.nonzero(self.ground_truth == 2)[0], np.nonzero(self.ground_truth == 2)[1]))
        idx = np.random.choice(len(start_cells), N_AGENTS, replace=False)
        chosen = start_cells[idx]
        rows, cols = chosen[:, 0], chosen[:, 1]
        world_x = self.belief_origin_x + cols * self.cell_size
        world_y = self.belief_origin_y + (H -1 - rows) * self.cell_size
        self.robot_locations = np.stack([world_x, world_y], axis=1)

        # 4) Initialize headings
        self.angles = np.random.uniform(0, 360, size=N_AGENTS)

        # 5) Prepare belief_info for coordinate conversions
        self.belief_info = MapInfo(self.robot_belief,
                                   self.belief_origin_x,
                                   self.belief_origin_y,
                                   self.cell_size)

        # 6) Perform initial sensing update for each agent
        for i in range(N_AGENTS):
            cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            
            self.robot_belief = sensor_work_heading(
                cell,
                round(self.sensor_range / self.cell_size),
                self.robot_belief,
                self.ground_truth,
                self.angles[i],
                360
            )

        # 7) Compute goal coordinates (average over all goal cells)
        goal_cells = np.column_stack(np.nonzero(self.ground_truth == 3))
        goal_world = []
        for row, col in goal_cells:
            x = self.belief_origin_x + col * self.cell_size
            y = self.belief_origin_y + (H - 1 - row) * self.cell_size
            goal_world.append([x, y])
        self.goal_coords = np.mean(goal_world, axis=0)

        self.reached_goal = [False] * N_AGENTS
        
        # 8) Optional plotting setup
        if self.plot:
            self.frame_files = []

    def import_ground_truth(self, episode_index):
        map_dir = '/home/dbtngud/myproject/utils/myMap'
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

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell, heading):
        self.robot_belief = sensor_work_heading(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth, heading, self.fov)


    def check_done(self) -> int:
        """
        Returns:
            status_code (int):
              0  = 계속 진행
              1  = 목표 도달
             -1  = 장애물 충돌 또는 에이전트 간 충돌
        Also sets self.done = (status_code != 0).
        """
        # 1) 로봇 셀 좌표
        cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        rows, cols = cells[:, 0], cells[:, 1]
        
        # 2) 종료 조건 판정
        in_goal      = np.all(self.ground_truth[rows, cols] == 3)
        hit_obstacle = np.any(self.ground_truth[rows, cols] == 1)
        # 에이전트 간 충돌 (동일 셀 공유)
        flat_idx  = rows * self.ground_truth.shape[1] + cols
        counts    = np.bincount(flat_idx)
        collision = np.any(counts > 1)

        # 3) 상태 코드 결정
        if in_goal:
            code = 1
        elif hit_obstacle or collision:
            code = -1
        else:
            code = 0

        # 4) self.done 설정
        self.done = (code != 0)

        # 5) 이유 출력
        if code == 1:
            print("[Done] 종료 사유: 목표 도달")
        elif code == -1:
            reason_list = []
            if hit_obstacle:
                reason_list.append("장애물 충돌")
            if collision:
                reason_list.append("에이전트 간 충돌")
            print(f"[Done] 종료 사유: {', '.join(reason_list)}")

        return code


    def step(self, actions, dt=0.1):
        n = len(actions)

        # 1) 이전 거리 저장
        prev_dists = [
            np.linalg.norm(self.robot_locations[i] - self.goal_coords)
            for i in range(n)
        ]
        

        # 2) 위치·헤딩 업데이트 + Belief 갱신
        for i, (v, yaw_rate) in enumerate(actions):
            # 이미 도달한 에이전트는 대기
            if self.reached_goal[i]:
                continue

            # a) 헤딩 갱신
            angle_new = (self.angles[i] + yaw_rate * dt) % 360
            self.angles[i] = angle_new

            # b) 위치 갱신
            dx = v * dt * np.cos(np.radians(angle_new))
            dy = v * dt * np.sin(np.radians(angle_new))
            self.robot_locations[i] += np.array([dx, dy])

            # c) Belief 업데이트
            cell = get_cell_position_from_coords(self.robot_locations[i], self.belief_info)
            self.update_robot_belief(cell, angle_new)
        
        # 3) 보상 계산 및 도달 플래그 설정
        reward_list = []

        # 먼저 각 에이전트의 셀 위치 한 번에 계산
        cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        rows, cols = cells[:, 1], cells[:, 0]

        # 전체 맵 판정
        in_goal_mask    = (self.ground_truth[rows, cols] == 3)
        hit_obstacle_mask = (self.ground_truth[rows, cols] == 1)

        for i in range(n):
            curr_dist   = np.linalg.norm(self.robot_locations[i] - self.goal_coords)
            dist_reward = prev_dists[i] - curr_dist
            reward      = dist_reward

            # goal 도달 시 첫 보너스
            if in_goal_mask[i]:
                if not self.reached_goal[i]:
                    self.reached_goal[i] = True
                    reward += REWARD_GOAL  # 첫 도달 보너스
                # 모두 도달 대기 로직 유지

            if hit_obstacle_mask[i]:
                reward -= 100.0

            reward_list.append(reward)

        done = self.check_done()
        info = {}

        self.evaluate_exploration_rate()

        return reward_list, done, info

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
        explored_cells = np.count_nonzero(belief != UNKNOWN)
        # Compute exploration rate
        self.explored_rate = explored_cells / total_cells
    
    def heading_to_vector(self, heading, length=1):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length

    def create_sensing_mask(self, location, heading):
        mask = np.zeros_like(self.ground_truth)

        location_cell = get_cell_position_from_coords(location, self.belief_info)
        # Create a Point for the robot's location
        robot_point = Point(location_cell)
        # heading = heading*(360/self.num_angles)

        # Calculate the angles for the sector
        start_angle = (heading - self.fov / 2 + 360) % 360
        end_angle = (heading + self.fov / 2) % 360

        # Create points for the sector
        sector_points = [robot_point]
        if start_angle <= end_angle:
            angle_range = np.linspace(start_angle, end_angle, 20)
        else:
            angle_range = np.concatenate([np.linspace(start_angle, 360, 10), np.linspace(0, end_angle, 10)])
        for angle in angle_range: 
            x = location_cell[0] + self.sensor_range/CELL_SIZE * np.cos(np.radians(angle))
            y = location_cell[1] + self.sensor_range/CELL_SIZE * np.sin(np.radians(angle))
            sector_points.append(Point(x, y))
        sector_points.append(robot_point) 

        sector = Polygon(sector_points)

        x_coords, y_coords = sector.exterior.xy
        y_coords = np.rint(y_coords).astype(int)
        x_coords = np.rint(x_coords).astype(int)
        rr, cc = sk_polygon(
                [int(round(y)) for y in y_coords],
                [int(round(x)) for x in x_coords],
                shape=mask.shape
            )
        
        free_connected_map = get_free_and_connected_map(location, self.belief_info)

        mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])
       
        return mask
    