import numpy as np
import imageio
import os
from skimage.morphology import label

from parameter import *


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    """
        coords: array-like of shape (2,) or (N,2) in world [x, y] coordinates
        map_info: MapInfo 객체 (map, map_origin_x, map_origin_y, cell_size)
        check_negative: True일 경우, boundary 밖으로 나간 인덱스를 자동으로 클램핑
    """
    # 1) 입력 형태 정리
    single_cell = False
    coords = np.asarray(coords)
    if coords.ndim == 1 and coords.size == 2:
        single_cell = True
        coords = coords.reshape(1, 2)
    else:
        coords = coords.reshape(-1, 2)

    # 2) world → fractional cell 좌표
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    H, W = map_info.map.shape

    # y 축 뒤집기: 맵 row 0이 하단(origin_y)에 대응하도록
    frac_x = (coords_x - map_info.map_origin_x) / map_info.cell_size
    frac_y = H - 1 - ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    # 3) 내림 처리하여 정수 인덱스로 변환
    cell_position = np.floor(
        np.stack((frac_x, frac_y), axis=-1)
    ).astype(int)

    # 4) boundary 클램핑 (optional)
    if check_negative:
        cell_position[:, 0] = np.clip(cell_position[:, 0], 0, W - 1)
        cell_position[:, 1] = np.clip(cell_position[:, 1], 0, H - 1)

    # 5) 반환 형태 맞추기
    if single_cell:
        return cell_position[0]
    return cell_position

def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    H = map_info.map.shape[0]
    cell_y = H - 1 - cell_position[:, 1]

    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == OCCUPIED:

        return coords[0]
    else:
        return coords

def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map

def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    for filename in frame_files[:-1]:
        os.remove(filename)

def make_gif_test(path, n, frame_files, rate, n_agents, fov, sensor_range):
    with imageio.get_writer('{}/{}_{}_{}_{}_explored_rate_{:.4g}.gif'.format(path, n, n_agents, fov, sensor_range, rate), mode='I', duration=1) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')
    for filename in frame_files[:-1]:
        os.remove(filename)



# class MapInfo:
#     def __init__(self, map, map_origin_x, map_origin_y, cell_size):
#         self.map = map
#         self.map_origin_x = map_origin_x
#         self.map_origin_y = map_origin_y
#         self.cell_size = cell_size

#     def update_map_info(self, map, map_origin_x, map_origin_y):
#         self.map = map
#         self.map_origin_x = map_origin_x
#         self.map_origin_y = map_origin_y


