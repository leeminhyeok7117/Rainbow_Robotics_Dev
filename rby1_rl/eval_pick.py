import argparse, glob, re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rby1_pick_env import Rby1PickEnv

p = argparse.ArgumentParser()
p.add_argument("--dir", default="runs/pick")
p.add_argument("--episodes", type=int, default=40)
p.add_argument("--reward-mode", choices=["sparse", "dense"], default="dense")
p.add_argument("--ladder", action="store_true")
a = p.parse_args()
mode = a.reward_mode
env = Rby1PickEnv(reward_mode=mode)

def evaluate(zip_path, pkl_path):
    vn = VecNormalize.load(pkl_path, DummyVecEnv([(lambda m=mode: Rby1PickEnv(reward_mode=m))]))
    vn.training = False
    m = PPO.load(zip_path, device="cpu")
    succ, heights, grasped = [], [], []
    for _ in range(a.episodes):
        o, _ = env.reset()
        max_h = -1
        for t in range(200):
            act, _ = m.predict(vn.normalize_obs(o), deterministic=True)
            o, r, te, tr, info = env.step(act)
            max_h = max(max_h, info["height"])
            grasped.append(info["grasped"] == 2)
        succ.append(info["is_success"]); heights.append(max_h)
    return (np.mean(succ) * 100, np.mean(heights) * 1000,
            np.mean(grasped) * 100)

if a.ladder:
    cks = sorted(glob.glob(f"{a.dir}/pick_*_steps.zip"),
                 key=lambda f: int(re.search(r"_(\d+)_steps", f).group(1)))
    print(f"[{mode}] {a.dir}")
    print(f"{'스텝':>9s} {'성공률':>7s} {'최대높이':>9s} {'그립비율':>9s}")
    print("-" * 40)
    for c in cks:
        n = int(re.search(r"_(\d+)_steps", c).group(1))
        s, h, g = evaluate(c, f"{a.dir}/pick_vecnormalize_{n}_steps.pkl")
        print(f"{n:9d} {s:6.0f}% {h:8.0f}mm {g:8.0f}%")
else:
    s, h, g = evaluate(f"{a.dir}/ppo_pick.zip", f"{a.dir}/vecnorm.pkl")
    print(f"[{mode}] 성공률(0.2s+ 8cm 이상 유지) {s:.0f}%   평균 최대높이 {h:.0f}mm   양손가락 그립 비율 {g:.0f}%")
