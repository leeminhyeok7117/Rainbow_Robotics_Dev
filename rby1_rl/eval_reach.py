import argparse
import numpy as np, imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rby1_reach_env import Rby1ReachEnv

p = argparse.ArgumentParser()
p.add_argument("--out", default="runs/reach")
p.add_argument("--episodes", type=int, default=50)
p.add_argument("--video", default="reach.mp4")
a = p.parse_args()

venv = DummyVecEnv([lambda: Rby1ReachEnv()])
venv = VecNormalize.load(f"{a.out}/vecnorm.pkl", venv)
venv.training, venv.norm_reward = False, False           # 평가 시 통계 갱신 중단
model = PPO.load(f"{a.out}/ppo_reach", device="cpu")

succ, dists, frames = [], [], []
for ep in range(a.episodes):
    obs = venv.reset()
    for t in range(150):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(act)
        if ep < 3 and t % 2 == 0:
            frames.append(venv.envs[0].render())
    succ.append(info[0]["is_success"]); dists.append(info[0]["dist"])

print(f"success rate {np.mean(succ)*100:.0f}%  "
      f"final dist mean {np.mean(dists)*1000:.0f}mm  median {np.median(dists)*1000:.0f}mm")
imageio.mimsave(a.video, frames, fps=25)
print("video ->", a.video)
