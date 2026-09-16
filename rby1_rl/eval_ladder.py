import argparse, glob, re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rby1_reach_env import Rby1ReachEnv

p = argparse.ArgumentParser()
p.add_argument("--dir", default="runs/ladder")
p.add_argument("--episodes", type=int, default=40)
p.add_argument("--torque", action="store_true")
a = p.parse_args()
mode = "torque" if a.torque else "position"

cks = sorted(glob.glob(f"{a.dir}/reach_*_steps.zip"),
             key=lambda f: int(re.search(r"_(\d+)_steps", f).group(1)))
env = Rby1ReachEnv(control_mode=mode)
print(f"[{mode}] {a.dir}")
print(f"{'스텝':>8s} {'성공률':>7s} {'평균오차':>9s} {'중앙값':>8s}")
print("-" * 38)
for c in cks:
    n = int(re.search(r"_(\d+)_steps", c).group(1))
    vn = VecNormalize.load(f"{a.dir}/reach_vecnormalize_{n}_steps.pkl",
                           DummyVecEnv([(lambda m=mode: Rby1ReachEnv(control_mode=m))]))
    vn.training = False
    m = PPO.load(c, device="cpu")
    s, d = [], []
    for _ in range(a.episodes):
        o, _ = env.reset()
        for t in range(150):
            act, _ = m.predict(vn.normalize_obs(o), deterministic=True)
            o, r, te, tr, info = env.step(act)
        s.append(info["is_success"]); d.append(info["dist"])
    print(f"{n:8d} {np.mean(s)*100:6.0f}% {np.mean(d)*1000:7.0f}mm {np.median(d)*1000:6.0f}mm")
