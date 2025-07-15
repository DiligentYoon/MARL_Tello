import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ray
import wandb
from torch.utils.tensorboard import SummaryWriter

from utils.model import MLPPolicy, MLPCritic
from utils.runner import RLRunner
from parameter import *

ray.init()
print("=== Training Start ===")
# TensorBoard / WandB
writer = SummaryWriter(train_path)
if USE_WANDB:
    wandb.init(project='MARVEL', name=FOLDER_NAME, config=vars(parameter))

# make dirs
for p in [model_path, gifs_path, load_path]:
    os.makedirs(p, exist_ok=True)

def main():
    # devices
    device       = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # 1) 네트워크 초기화 (Q 하나만)
    device     = torch.device('cuda' if USE_GPU_GLOBAL else 'cpu')
    policy_net = MLPPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM).to(device)
    q_net      = MLPCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM).to(device)
    target_q   = MLPCritic(obs_dim=OBS_DIM, action_dim=ACTION_DIM).to(device)
    target_q.load_state_dict(q_net.state_dict())
    target_q.eval()

    # 2) Optimizer & α
    opt_pi    = optim.Adam(policy_net.parameters(), lr=LR)
    opt_q     = optim.Adam(q_net.parameters(),      lr=LR)
    log_alpha = torch.tensor(-2.0, requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=LR)
    target_entropy = -ACTION_DIM

    def soft_update(tgt, src, tau):
        for t, s in zip(tgt.parameters(), src.parameters()):
            t.data.mul_(1-tau)
            t.data.add_(s.data * tau)

    # 3) Replay buffer & workers & perf 기록
    buffer  = [[] for _ in range(NUM_EPISODE_BUFFER)]  # obs, act, rew, next_obs, done
    perf    = {'travel_dist': [], 'success_rate': [], 'explored_rate': []}
    workers = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]
    weights = [policy_net.state_dict()]
    jobs    = [w.job.remote(weights, i+1) for i,w in enumerate(workers)]
    episode = 1 + len(workers)

    try:
        while True:
            done_id, jobs = ray.wait(jobs)
            results, metrics, info = ray.get(done_id[0])

            # collect experiences
            for i in range(NUM_EPISODE_BUFFER):
                buffer[i].extend(results[i])
            for k in perf:
                perf[k].append(metrics[k])

            # launch next
            jobs.append(workers[info['id']].job.remote(weights, episode))
            episode += 1

            # start SAC training once enough
            if len(buffer[0]) >= MIN_BUFFER_SIZE:
                # a) 버퍼 trimming
                for k in range(NUM_EPISODE_BUFFER):
                    if len(buffer[k]) > REPLAY_SIZE:
                        buffer[k] = buffer[k][-REPLAY_SIZE:]

                idxs = random.sample(range(len(buffer[0])), BATCH_SIZE)
                obs_b      = torch.stack([buffer[0][i] for i in idxs]).to(device)
                act_b      = torch.stack([buffer[1][i] for i in idxs]).to(device)
                rew_b      = torch.tensor([buffer[2][i] for i in idxs],dtype=torch.float32, device=device).unsqueeze(1)
                next_obs_b = torch.stack([buffer[3][i] for i in idxs]).to(device)
                done_b     = torch.tensor([buffer[4][i] for i in idxs],dtype=torch.float32, device=device).unsqueeze(1)

                # b) Critic loss
                with torch.no_grad():
                    mu2, log_std2 = policy_net(next_obs_b)
                    std2          = log_std2.exp()
                    dist2         = torch.distributions.Normal(mu2, std2)
                    a2            = dist2.rsample()
                    logp2         = dist2.log_prob(a2).sum(-1, keepdim=True)
                    q_next        = target_q(next_obs_b, a2)
                    target_q_val  = rew_b + GAMMA * (1 - done_b) * (q_next - log_alpha.exp() * logp2)

                q_pred = q_net(obs_b, act_b)
                loss_q = nn.MSELoss()(q_pred, target_q_val)
                opt_q.zero_grad(); loss_q.backward(); opt_q.step()

                # c) Actor loss
                mu, log_std = policy_net(obs_b)
                std         = log_std.exp()
                dist        = torch.distributions.Normal(mu, std)
                a_sample    = dist.rsample()
                logp        = dist.log_prob(a_sample).sum(-1, keepdim=True)
                q_pi        = q_net(obs_b, a_sample)
                loss_pi     = (log_alpha.exp() * logp - q_pi).mean()
                opt_pi.zero_grad();    loss_pi.backward(); opt_pi.step()

                # d) Alpha loss
                loss_alpha = -(log_alpha * (logp + target_entropy).detach()).mean()
                opt_alpha.zero_grad(); loss_alpha.backward(); opt_alpha.step()

                # e) Soft update
                soft_update(target_q, q_net, TAU)

                # f) Logging
                avg_perf = {k: np.mean(perf[k]) for k in perf}
                writer.add_scalar('Loss/Q',        loss_q.item(),    episode)
                writer.add_scalar('Loss/Policy',   loss_pi.item(),   episode)
                writer.add_scalar('Loss/Alpha',    loss_alpha.item(),episode)
                writer.add_scalar('Perf/TravelDist',   avg_perf['travel_dist'], episode)
                writer.add_scalar('Perf/SuccessRate',  avg_perf['success_rate'], episode)
                writer.add_scalar('Perf/ExploredRate', avg_perf['explored_rate'], episode)

                # broadcast new policy weights
                weights = [policy_net.state_dict()]

                # h) checkpoint
                if episode % SAVE_MODEL_GAP == 0:
                    torch.save({
                        'policy': policy_net.state_dict(),
                        'q':      q_net.state_dict(),
                        'log_alpha': log_alpha.data,
                        'episode': episode
                    }, os.path.join(model_path, 'checkpoint.pth'))
                    print(f"[Episode {episode}] Checkpoint saved")

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        for w in workers:
            ray.kill(w)
        ray.shutdown()
        writer.close()
        if USE_WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()
