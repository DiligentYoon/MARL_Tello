import os
from abc import abstractmethod
import datetime

import collections
import torch

from typing import Mapping, Optional
from torch.nn import Module
from utils.utils import *

class Agent:
    """
    Base Agent with the following requirements:
        1. 데이터 로깅 (Tensorboard)
        2. 액션 추출 (From Policy Network)
        3. Checkpoint 저장 및 세이브
        4. Checkpoint 로드
        5. 옵티마이저, 네트워크 초기화
    """
    def __init__(self,
                 id: int,
                 models: Mapping[str, Module],
                 device: torch.device,
                 cfg: Optional[dict] = None):
        
        self.id = id
        self.models = models
        self.cfg = cfg if cfg is not None else {}
        self.device = device

        # 모델 초기화
        for model in self.models.values():
            if model is not None:
                model.to(self.device)

        # 데이터 로깅 인스턴스 초기화 (Tensorboard)
        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", "auto")

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        # checkpoint
        self.checkpoint_modules = {}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", "auto")
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)

        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = "{}_{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"), self.__class__.__name__
            )
        self.experiment_dir = os.path.join(directory, experiment_name)



    def _get_internal_value(self, module: Module) -> dict:
        """
            모듈의 state_dict를 반환하는 헬퍼 함수 -> checkpoint 저장 등에 사용
        """
        return module.state_dict() if hasattr(module, "state_dict") else module


    def init(self, timesteps) -> None:
        # main entry to log data for consumption and visualization by TensorBoard -> Multi Agent로 보내기.
        # if self.write_interval == "auto":
        #     self.write_interval = int(self.cfg.get("timesteps", 0) / 100)
        # if self.write_interval > 0:
        #     self.writer = SummaryWriter(log_dir=self.experiment_dir)

        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(timesteps / 10)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)


    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        
        tag = str(timestep)
        
        # 개별 모듈로 저장하는 경우
        if self.checkpoint_store_separately:
            for name, module in self.checkpoint_modules.items():
                torch.save(
                    self._get_internal_value(module),
                    os.path.join(self.experiment_dir, "checkpoints", f"{name}_{tag}_{self.id}.pt"),
                )
        # 전체 에이전트를 하나의 파일로 저장하는 경우
        else:
            modules = {name: self._get_internal_value(module) for name, module in self.checkpoint_modules.items()}
            torch.save(modules, os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}_{self.id}.pt"))


    def post_interaction(self, timestep: int, timesteps: int) -> None:
        # 파이썬 인덱스는 0부터 시작하므로 1을 더해줌
        current_timestep = timestep + 1
        # 주기적 체크포인트 저장 함수 호출
        self.write_checkpoint(current_timestep, timesteps)


    def load_checkpoint(self, path: str):

        modules = torch.load(path, map_location=self.device)

        if type(modules) is dict:
            for name, data in modules.items():
                # self.checkpoint_modules에 해당 이름의 모듈이 있는지 확인
                module_to_load = self.checkpoint_modules.get(name)

                if module_to_load is not None:
                    # state_dict를 로드
                    module_to_load.load_state_dict(data)
                    print(f"Module '{name}' loaded from checkpoint.")
                else:
                    print(f"Warning: Module '{name}' found in checkpoint but not in agent.")
        else:
            print("Error: Checkpoint file is not a valid dictionary.")



    @abstractmethod
    def act(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Please implement the 'act' method for {self.__class__.__name__}.")
    
    @abstractmethod
    def update(self) -> None:
        raise NotImplementedError(f"Please implement the 'update' method for {self.__class__.__name__}.")