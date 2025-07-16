
from .env_cfg import EnvCfg


class APFEnvCfg(EnvCfg):
    num_obs: int
    num_act: int
    num_state: int
    decimation: int
    max_episode_steps: int
    apf_k_att: float
    apf_k_rep: float
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Space Information
        self.num_obs = 11
        self.num_act = 2
        self.num_state = 11

        # Episode Information
        self.decimation = 5
        self.max_episode_steps = 128

        # APF Information
        self.apf_k_att = 1.0
        self.apf_k_rep = 0.5

        # Local Patch Information
        self.patch_size = 30
        

        # Reward Hyperparameter (TBD)
