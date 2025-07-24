
from utils.base.env.env_cfg import EnvCfg


class APFActEnvCfg(EnvCfg):
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
        self.num_act = 3
        self.num_state = 11

        # Episode Information
        self.decimation = 1
        self.max_episode_steps = 256

        # APF Information
        self.apf_k_att_range = [0.0, 0.1]
        self.apf_k_rep_range = [0.0, 0.2]
        self.apf_k_inter_rep_range = [0.0, 0.25]
        self.rho = 0.1
        self.rho_agent = 0.25
        self.kp = 5.0

        # Local Patch Information
        self.patch_size = 30
        

        # Reward Hyperparameter (TBD)
