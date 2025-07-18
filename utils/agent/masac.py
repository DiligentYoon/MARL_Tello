import copy
import torch
import torch.nn.functional as F

from typing import Mapping, Optional, Any, Dict, Tuple
from torch.nn import Module

from utils.base.agent.multi_agent import MultiAgent

class MASACAgent(MultiAgent):
    """
    Multi-Agent Soft Actor-Critic (SAC) Agent with Centralized Critics.
    This agent implements the BaseMultiAgent interface and is designed to be used
    within the MainDriver-RolloutWorker architecture.
    """
    def __init__(self, 
                 num_agents: int,
                 models: Mapping[str, Module], 
                 device: torch.device, 
                 cfg: Optional[dict] = None):
        
        super().__init__(num_agents, models, device, cfg)

        # --- Model Registration ---
        self.policy = self.models.get("policy")
        self.critic_1 = self.models.get("critic_1")
        self.critic_2 = self.models.get("critic_2")
        
        # Create target networks and sync their weights
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        # Freeze target networks - they are only updated via polyak averaging
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

        # --- Hyperparameters ---
        self.discount_factor = self.cfg.get("discount_factor", 0.99)
        self.polyak = self.cfg.get("polyak", 0.005)
        self.grad_norm_clip = self.cfg.get("grad_norm_clip", 0.5)
        self.lr = self.cfg.get("learning_rate", 1e-3)
        self.entropy_learning_rate = self.cfg.get("entropy_learning_rate", 1e-3)

        if type(self.lr) == str:
            self.lr = float(self.lr)

        if type(self.entropy_learning_rate) == str:
            self.entropy_learning_rate = float(self.entropy_learning_rate)


        # --- Optimizers ---
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.lr
        )

        # --- Entropy Tuning ---
        self.learn_entropy = self.cfg.get("learn_entropy", True)
        if self.learn_entropy:
            self.target_entropy = self.cfg.get("target_entropy", -self.models['policy'].log_std_layer.out_features) # Heuristic
            self.log_alpha = torch.tensor(self.cfg.get("initial_entropy_value", 0.2), device=self.device).log().requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.entropy_learning_rate)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(self.cfg.get("initial_entropy_value", 0.2), device=self.device)

        # --- Register modules for checkpointing by the MainDriver ---
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
        self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
        if self.learn_entropy:
            self.checkpoint_modules["log_alpha"] = self.log_alpha
            self.checkpoint_modules["alpha_optimizer"] = self.alpha_optimizer

    def act(self, states: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by the RolloutWorker. Samples actions from the policy.
        If deterministic is True, it returns the mean of the policy distribution.

        :param states: A tensor of shape (num_agents, obs_dim).
        :param deterministic: Whether to sample from the distribution or take the mean.
        :return: A tensor of shape (num_agents, action_dim).
        """
        self.policy.eval()
        with torch.no_grad():
            if deterministic:
                # For evaluation, take the mean of the distribution
                mu, logp = self.policy(states)
                # SAC actions are squashed by tanh, so the deterministic action should also be squashed.
                actions = torch.tanh(mu)
            else:
                # For training, sample from the distribution
                # The `sample` method from ActorGaussianNet already returns tanh-squashed actions
                actions, logp = self.policy.compute(states)
        self.policy.train()

        return actions, logp

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Called by the MainDriver. Performs one step of SAC update.
        Assumes the batch is a dictionary of tensors sampled from a replay buffer,
        where each entry corresponds to a single agent's transition.
        """
        # Unpack batch from dictionary and move to device
        obs = batch['obs'].to(self.device)
        state = batch['state'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        dones = (batch['terminated'] | batch['truncated']).to(self.device)

        batch_size = next_obs.size(0)
        num_agents = self.num_agents

        # --- Critic Update ---
        with torch.no_grad():
            # Get next actions and log probs from the decentralized policy
            # (B, obs_dim) -> (B, act_dim)
            next_actions, next_log_pi = self.policy.compute(next_obs)

            # Flatten the joint actions to concatenate with the state for the critic
            # (B, (state_dim + act_dim))
            critic_input_next = torch.cat([next_state, next_actions], dim=1)

            # (B, (state_dim + act_dim)) -> (B, 1)
            q1_next_target = self.target_critic_1.compute(critic_input_next)
            q2_next_target = self.target_critic_2.compute(critic_input_next)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_pi
            next_q_value = rewards + (~dones) * self.discount_factor * min_q_next_target

        # Get current Q estimates
        # (B, (state_dim + act_dim)) -> (B, 1)
        critic_input_current = torch.cat([state, actions], dim=1)
        q1_current = self.critic_1.compute(critic_input_current)
        q2_current = self.critic_2.compute(critic_input_current)
        
        critic_loss = (F.mse_loss(q1_current, next_q_value) + F.mse_loss(q2_current, next_q_value)) / 2

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), self.grad_norm_clip)
        self.critic_optimizer.step()

        # --- Policy and Alpha Update ---
        for p in self.critic_1.parameters(): p.requires_grad = False
        for p in self.critic_2.parameters(): p.requires_grad = False

        # The policy is updated using decentralized observations
        # (B, obs_dim) -> (B, act_dim)
        pi, log_pi = self.policy.compute(obs)
        
        # Policy Loss Calculation
        q1_pi = self.critic_1(torch.cat([state, pi], dim=1))
        q2_pi = self.critic_2(torch.cat([state, pi], dim=1))
        policy_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_clip)
        self.policy_optimizer.step()

        for p in self.critic_1.parameters(): p.requires_grad = True
        for p in self.critic_2.parameters(): p.requires_grad = True

        if self.learn_entropy:
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.mul_(1.0 - self.polyak)
                target_param.data.add_(param.data * self.polyak)
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.mul_(1.0 - self.polyak)
                target_param.data.add_(param.data * self.polyak)

        return {
            "loss/critic": critic_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "alpha": self.alpha.item()
        }
