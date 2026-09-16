import argparse, glob, os, re, sys, time
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rby1_pick_env import Rby1PickEnv

p = argparse.ArgumentParser()
p.add_argument("--out", default="runs/pick_dense")
p.add_argument("--reward-mode", choices=["sparse", "dense"], default="dense")
p.add_argument("--ckpt", type=int, default=None,
               help="중간 체크포인트 스텝 수 (예: 400000). 생략하면 최종 정책")
p.add_argument("--list", action="store_true")
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--stochastic", action="store_true")
p.add_argument("--seed", type=int, default=None)
a = p.parse_args()
mode = a.reward_mode


def available(d):
    return sorted(int(re.search(r"_(\d+)_steps", f).group(1))
                  for f in glob.glob(f"{d}/pick_*_steps.zip"))


if a.list:
    ck = available(a.out)
    print(f"{a.out} 체크포인트: {ck if ck else '없음'}")
    if os.path.exists(f"{a.out}/ppo_pick.zip"):
        print(f"{a.out} 최종 정책: ppo_pick.zip")
    sys.exit(0)

if a.ckpt is not None:
    zip_path = f"{a.out}/pick_{a.ckpt}_steps.zip"
    pkl_path = f"{a.out}/pick_vecnormalize_{a.ckpt}_steps.pkl"
    if not os.path.exists(zip_path):
        ck = available(a.out)
        sys.exit(f"체크포인트 없음: {zip_path}\n사용 가능: {ck if ck else '없음 (--list 로 확인)'}")
    label = f"{a.ckpt//1000}k steps"
else:
    zip_path, pkl_path = f"{a.out}/ppo_pick", f"{a.out}/vecnorm.pkl"
    label = "final"

vn = VecNormalize.load(pkl_path, DummyVecEnv([(lambda m=mode: Rby1PickEnv(reward_mode=m))]))
vn.training = False
model = PPO.load(zip_path, device="cpu")
print(f"loaded [{mode}/{label}] from {a.out}")

env = Rby1PickEnv(reward_mode=mode)
dt = env.frame_skip * env.model.opt.timestep / a.speed
ep = 0
obs, _ = env.reset(seed=a.seed)
succ = []

with mujoco.viewer.launch_passive(env.model, env.data) as v:
    next_t = time.time()
    while v.is_running():
        act, _ = model.predict(vn.normalize_obs(obs), deterministic=not a.stochastic)
        obs, r, term, trunc, info = env.step(act)
        v.sync()
        if trunc:
            succ.append(info["is_success"])
            print(f"[{label}] ep {ep:3d}  최대높이 {info['height']*1000:5.1f}mm  "
                  f"그립 {info['grasped']}/2  "
                  f"{'SUCCESS' if info['is_success'] else 'fail   '}  "
                  f"(누적 성공률 {np.mean(succ)*100:3.0f}%, n={len(succ)})")
            ep += 1
            obs, _ = env.reset(seed=None if a.seed is None else a.seed + ep)
            time.sleep(0.4)
            next_t = time.time()
        next_t += dt
        time.sleep(max(0.0, next_t - time.time()))
