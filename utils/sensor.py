import numpy as np
from parameter import *
from skimage.draw import line

def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    """
    Ray-cast from (x0,y0) to (x1,y1) in cell coordinates.
    Update robot_belief cell-by-cell:
      - FREE 영역은 FREE로,
      - 첫 번째 OCCUPIED 셀은 OCCUPIED로 표시한 뒤 중단,
      - 그 이후는 UNKNOWN(기존 상태 유지).
    """
    # 1) 정수 셀 인덱스로 변환
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))

    # 2) Bresenham 준비
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    # 3) 레이 캐스팅 루프
    while True:
        # 3.1) 맵 범위 체크
        if not (0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]):
            break

        # 3.2) 셀 클래스 읽기
        gt = ground_truth[y, x]

        if gt == 1:
            # 충돌 지점만 OCCUPIED로 업데이트하고 종료
            robot_belief[y, x] = OCCUPIED
            break
        else:
            # FREE 또는 기타(UNKNOWN) 영역은 FREE로 업데이트
            robot_belief[y, x] = FREE

        # 3.3) 종료 조건: 끝점 도달
        if x == x1 and y == y1:
            break

        # 3.4) Bresenham step
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 <  dx:
            err += dx
            y   += sy

    return robot_belief


def sensor_work(robot_position, sensor_range, robot_belief, ground_truth):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief

def normalize_angle(angle):
    """Normalize an angle to be within [0, 360) degrees."""
    return angle % 360

def calculate_fov_boundaries(center_angle, fov):
    """Calculate the start and end angles of the field of vision (FOV).
    
    Args:
        center_angle (float): The central angle of the FOV in degrees.
        fov (float): The total field of vision in degrees.
        
    Returns:
        (float, float): The start and end angles of the FOV.
    """
    half_fov = fov / 2
    
    start_angle = center_angle - half_fov
    end_angle = center_angle + half_fov
    
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)
    
    return start_angle, end_angle

def fov_sweep(start_angle, end_angle, increment):
    """Generate the correct sequence of angles to sweep the FOV from start to end with a specified increment.
    
    Args:
        start_angle (float): The starting angle of the FOV in degrees.
        end_angle (float): The ending angle of the FOV in degrees.
        increment (float): The angle increment in degrees.
        
    Returns:
        list: The sequence of angles representing the FOV sweep.
    """
    angles = []
    
    if start_angle < end_angle:
        angles = list(np.arange(start_angle, end_angle + increment, increment))
    else:
        angles = list(np.arange(start_angle, 360, increment)) + list(np.arange(0, end_angle + increment, increment))
    
    angles = [angle % 360 for angle in angles]
    
    angles_in_radians = np.radians(angles)

    return angles_in_radians

def sensor_work_heading(robot_position, sensor_range, robot_belief, ground_truth, heading, fov):

    sensor_angle_inc = 0.5
    x0 = robot_position[0]
    y0 = robot_position[1]
    start_angle, end_angle = calculate_fov_boundaries(heading, fov)
    sweep_angles = fov_sweep(start_angle, end_angle, sensor_angle_inc)

    x1_values = []
    y1_values = []
    
    for angle in sweep_angles:
        x1 = x0 + np.cos(angle) * sensor_range    
        y1 = y0 + np.sin(-angle) * sensor_range
        x1_values.append(x1)
        y1_values.append(y1)    
        
        # # 5) Debug prints with f-strings
        # print(f"[sensor] origin: ({x0:.2f}, {y0:.2f}), "
        #       f"end: ({x1:.2f}, {y1:.2f}), angle: {np.degrees(angle):.1f}°")

        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)

    return robot_belief