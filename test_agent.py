import yaml
import torch
from utils.model.model import ActorGaussianNet, CriticDeterministicNet
from utils.agent.masac import MASACAgent

def test_masac_agent():
    """
    A simple test bench to verify the initialization and single-step functionality
    of the MASACAgent.
    """
    print("--- Starting MASACAgent Test ---")

    # 1. Load Configuration
    try:
        with open("sac_cfg.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        print("[SUCCESS] Configuration loaded from sac_cfg.yaml")
    except FileNotFoundError:
        print("[ERROR] sac_cfg.yaml not found. Please ensure the file is in the root directory.")
        return

    # --- Test Parameters ---
    env_cfg = cfg.get("env", {})
    agent_cfg = cfg.get("agent", {})
    model_cfg = cfg.get("model", {})

    num_agents = env_cfg.get("num_agent", 4)
    batch_size = agent_cfg.get("batch_size", 64)
    # Assuming individual agent observation and action dimensions
    # For centralized critic, these will be combined later.
    obs_dim = 10  # Placeholder, replace with your actual obs_dim
    action_dim = 2 # Placeholder, replace with your actual action_dim
    state_dim = 12
    device = torch.device(cfg.get("device", "cpu"))

    print(f"Device: {device}, Num Agents: {num_agents}")

    # 2. Initialize Models
    try:
        policy = ActorGaussianNet(obs_dim, action_dim, device, model_cfg['actor'])
        
        # Centralized critic takes concatenated obs and actions from all agents
        critic_obs_dim = state_dim + action_dim
        critic_action_dim = 1
        
        critic1 = CriticDeterministicNet(critic_obs_dim, critic_action_dim, device, model_cfg['critic'])
        critic2 = CriticDeterministicNet(critic_obs_dim, critic_action_dim, device, model_cfg['critic'])
        
        models = {
            "policy": policy,
            "critic_1": critic1,
            "critic_2": critic2
        }
        print("[SUCCESS] Actor and Critic models initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize models: {e}")
        return

    # 3. Initialize MASACAgent
    try:
        agent = MASACAgent(num_agents=num_agents, models=models, device=device, cfg=agent_cfg)
        print("[SUCCESS] MASACAgent initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize MASACAgent: {e}")
        return

    # 4. Test act() method
    try:
        # Create a dummy observation tensor: (num_agents, obs_dim)
        dummy_obs = torch.randn(num_agents, obs_dim).to(device)
        
        # Test stochastic action sampling (for training)
        stochastic_actions, _ = agent.act(dummy_obs, deterministic=False)
        assert stochastic_actions.shape == (num_agents, action_dim), "Stochastic action shape mismatch"
        print(f"[SUCCESS] act(deterministic=False) returned correct shape: {stochastic_actions.shape}")

        # Test deterministic action sampling (for evaluation)
        deterministic_actions, _ = agent.act(dummy_obs, deterministic=True)
        assert deterministic_actions.shape == (num_agents, action_dim), "Deterministic action shape mismatch"
        print(f"[SUCCESS] act(deterministic=True) returned correct shape: {deterministic_actions.shape}")

    except Exception as e:
        print(f"[ERROR] act() method failed: {e}")
        return

    # 5. Test update() method
    try:
        # Create a mock batch of data (1 transition for simplicity)
        dummy_state = torch.randn(batch_size, state_dim)
        dummy_next_state = torch.randn(batch_size, state_dim)
        dummy_action = torch.randn(batch_size, action_dim)
        dummy_obs = torch.randn(batch_size, obs_dim)
        dummy_next_obs = torch.randn(batch_size, obs_dim)
        dummy_rewards = torch.randn(batch_size, 1) # Assuming team-based reward for simplicity
        dummy_terminated = torch.zeros(batch_size, 1, dtype=torch.bool)
        dummy_truncated = torch.zeros(batch_size, 1, dtype=torch.bool)

        mock_batch = {
            'obs': dummy_obs,
            'state': dummy_state,
            'actions': dummy_action,
            'rewards': dummy_rewards,
            'next_obs': dummy_next_obs,
            'next_state': dummy_next_state,
            'terminated': dummy_terminated,
            'truncated': dummy_truncated
        }
    
        loss_dict = agent.update(mock_batch)
        print(f"[SUCCESS] update() method executed. Losses: {loss_dict}")
        assert "loss/critic" in loss_dict, "Critic loss not found in return dict"
        assert "loss/policy" in loss_dict, "Policy loss not found in return dict"

    except Exception as e:
        print(f"[ERROR] update() method failed: {e}")
        return

    print("--- MASACAgent Test Finished Successfully ---")

if __name__ == '__main__':
    test_masac_agent()
