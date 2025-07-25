import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ActorGaussianNet(nn.Module):
    def __init__(self, obs_dim, action_dim, device, cfg):
        super(ActorGaussianNet, self).__init__()
        self.cfg = cfg
        self.device = device
        in_dim = obs_dim
        hidden_sizes = self.cfg["hidden"]
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mu_layer    = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            mu:    (batch, action_dim)
            log_std: (batch, action_dim), clipped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    

    def compute(self, obs: torch.Tensor) -> torch.Tensor:
        """
            Samples an action from the policy, using reparameterization trick.
            Returns:
                action: (batch, action_dim)
                logp:   (batch, 1) log probability of sampled action
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # correction for Tanh squashing
        logp = dist.log_prob(z).sum(-1, keepdim=True) \
               - (2*(math.log(2) - z - F.softplus(-2*z))).sum(-1, keepdim=True)
        
        return action, logp


class CriticDeterministicNet(nn.Module):
    """
    Centralized Q‐value network.
    Inputs: joint_obs = [agent1_obs, ..., agentN_obs] 
            joint_act = [agent1_act, ..., agentN_act]
    Output: scalar Q
    """
    def __init__(self, obs_dim, action_dim, device, cfg):
        super(CriticDeterministicNet, self ).__init__()
        self.input_dim = obs_dim
        self.device = device
        in_dim = self.input_dim
        hidden_sizes = cfg["hidden"]
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.q_out = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return self.q_out(x)
    

    def compute(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)
