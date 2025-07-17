
import torch
import itertools
import copy
import numpy as np

from torch.nn import Module
from typing import Any, Mapping, Optional, Tuple, Union


from utils.base.agent.agent import Agent

SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.2,   # initial entropy value
    "target_entropy": None,         # target entropy

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately
    }
}

class SAC(Agent):
    def __init__(self,
                 id,
                 observation_space: np.ndarray,
                 action_space: np.ndarray,
                 models: Mapping[str, Module],
                 device: torch.device,
                 cfg: Optional[dict] = None) -> None:
        

        _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        self.observation_space = observation_space
        self.action_space = action_space

        super().__init__(id=id, models=models, device=device, cfg=_cfg)

        self.policy = self.models.get("actor")
        self.value = self.models.get("critic")

        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._device_type = torch.device(device).type
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
            self.checkpoint_modules["optimizer"] = self.optimizer

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer



    def act(self, state: torch.Tensor, deterministic= False) -> torch.Tensor:
        """
            액션 추출 (From Policy Network)
        """
        if self.policy is None:
            raise ValueError("Policy network ('policy') not found in models.")
        
        self.policy.eval() # Set to evaluation mode
    
        # sample stochastic actions
        with torch.autocast(device_type=self._device_type):
            mean, log_std = self.policy(state.to(self.device))
            self._current_log_prob = log_std
            std = log_std.exp()

            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        self.policy.train() # Set back to training mode

        return action.cpu()