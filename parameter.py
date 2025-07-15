"""
Configuration parameters for MARVEL simulation and training.

This module defines key parameters for:
- Folder and path configurations
- Simulation settings
- Drone and sensor characteristics 
- Map representation
- Training hyperparameters
- Neural network architecture
- Computational resource settings

Key configurations include:
- Number of agents
- Sensor ranges
- Map resolution
- Episode and training parameters
- GPU and logging options
"""

FOLDER_NAME = 'test3'
LOAD_FOLDER_NAME = 'joint_action_5_9_GT_MAAC'
model_path = f'model/{FOLDER_NAME}'
load_path = f'load_model/{LOAD_FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 32
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 1000
NUM_EPISODE_BUFFER = 40

# Sim parameters
N_AGENTS = 4
NUM_SIM_STEPS = 6
MAX_VELOCITY = 1
MAX_YAW_RATE = 35 # in degrees

# Heading parameters
FOV = 120   # in degrees

PATCH_SIZE = 30

# map and planning resolution
CELL_SIZE = 0.01  # meter


# belief map representation
FREE = 0
OCCUPIED = 2
UNKNOWN = 1

#Reward
GOAL_THRESHOLD = 0.2
REWARD_GOAL = 100

# sensor and utility range
SENSOR_RANGE = 0.3  # meter
UTILITY_RANGE = 0.9 * SENSOR_RANGE

# training parameters
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 256
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 18
ENTROPY_COEF    = 0.2    # 정책 엔트로피 가중치 (예시)
TAU             = 0.005  # 타깃 네트워크 소프트 업데이트 비율
SAVE_MODEL_GAP  = 1000   # 몇 에피소드마다 모델 저장할지


# === APF 계수 ===
K_ATT = 1.0   # 목표 유도력 계수
K_REP = 0.5   # 장애물 반발력 계수


# network parameters
OBS_DIM = 11    # [APF_x, APF_y,
                #  pos_x, pos_y, heading,
                #  vel_x, vel_y,
                #  nbr_pos_x, nbr_pos_y, nbr_vel_x, nbr_vel_y]

EMBEDDING_DIM = 128
ACTION_DIM = 2  # [velocity, yaw_rate]


# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1

USE_WANDB = False
TRAIN_ALGO = 3
# 0: SAC, 1:MAAC , 2: Ground Truth 3: MAAC and Ground Truth
