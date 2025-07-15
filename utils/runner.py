import torch
import ray

from utils.model import MLPPolicy  
from utils.multi_agent_worker import MultiAgentWorker
from parameter import *

class Runner:
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        # GPU 사용 여부
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.network = MLPPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        self.network.to(self.device)

    def get_weights(self):
        return self.network.state_dict()

    def set_policy_net_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        # 가끔 GIF 저장 여부 결정
        save_img = (episode_number % SAVE_IMG_GAP == 0)
        worker = MultiAgentWorker(
            self.meta_agent_id,
            self.network,
            episode_number,
            device=self.device,
            save_image=save_img
        )
        worker.run_episode()
        return worker.episode_buffer, worker.perf_metrics

    def job(self, weights_set, episode_number):
        print(f"[MetaAgent {self.meta_agent_id}] Episode {episode_number} 시작")
        # 새로운 파라미터 로드
        self.set_policy_net_weights(weights_set[0])
        results, metrics = self.do_job(episode_number)
        info = {"id": self.meta_agent_id, "episode_number": episode_number}
        return results, metrics, info


if USE_GPU:
    gpu_frac = NUM_GPU / NUM_META_AGENT
else:
    gpu_frac = 0

@ray.remote(num_cpus=1, num_gpus=gpu_frac)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)

if __name__ == "__main__":
    ray.init()
    # Runner 원격 액터 생성
    runner = RLRunner.remote(0)

    # 초기 weights 세팅
    init_policy = MLPPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
    init_weights = init_policy.state_dict()

    # Episode 1 실행
    results, metrics, info = ray.get(runner.job.remote([init_weights], 1))
    print("Metrics:", metrics)