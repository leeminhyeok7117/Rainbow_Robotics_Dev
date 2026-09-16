import argparse, glob, re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rby1_track_env import Rby1TrackEnv

p = argparse.ArgumentParser()
p.add_argument("--dir", default="runs/track")
p.add_argument("--episodes", type=int, default=40)
p.add_argument("--ladder", action="store_true", help="dir 안의 체크포인트를 전부 순회")
a = p.parse_args()

env = Rby1TrackEnv()

def evaluate(zip_path, pkl_path):
    vn = VecNormalize.load(pkl_path, DummyVecEnv([lambda: Rby1TrackEnv()]))
    vn.training = False
    m = PPO.load(zip_path, device="cpu")
    ratios, dists = [], []
    for _ in range(a.episodes):
        o, _ = env.reset()
        for t in range(150):
            act, _ = m.predict(vn.normalize_obs(o), deterministic=True)
            o, r, te, tr, info = env.step(act)
            dists.append(info["dist"])
        ratios.append(info["track_ratio"])
    return np.mean(ratios) * 100, np.mean(dists) * 1000, np.median(dists) * 1000

if a.ladder:
    cks = sorted(glob.glob(f"{a.dir}/track_*_steps.zip"),
                 key=lambda f: int(re.search(r"_(\d+)_steps", f).group(1)))
    print(f"{'스텝':>8s} {'추종비율':>8s} {'평균오차':>9s} {'중앙값':>8s}")
    print("-" * 38)
    for c in cks:
        n = int(re.search(r"_(\d+)_steps", c).group(1))
        tr, dm, dmed = evaluate(c, f"{a.dir}/track_vecnormalize_{n}_steps.pkl")
        print(f"{n:8d} {tr:7.0f}% {dm:8.0f}mm {dmed:7.0f}mm")
else:
    tr, dm, dmed = evaluate(f"{a.dir}/ppo_track.zip", f"{a.dir}/vecnorm.pkl")
    print(f"추종 비율(6cm 이내 시간) {tr:.0f}%   평균 오차 {dm:.0f}mm   중앙값 {dmed:.0f}mm")
