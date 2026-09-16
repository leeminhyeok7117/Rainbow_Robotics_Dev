import argparse, glob, os, re, sys, time
import numpy as np, mujoco, mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rby1_reach_env import Rby1ReachEnv

p = argparse.ArgumentParser()
p.add_argument("--out", default="runs/reach")
p.add_argument("--ckpt", type=int, default=None,
               help="중간 체크포인트 스텝 수 (예: 75000). 생략하면 최종 정책")
p.add_argument("--list", action="store_true", help="사용 가능한 체크포인트만 출력하고 종료")
p.add_argument("--speed", type=float, default=1.0)      # 0.5=슬로우모션, 2.0=2배속
p.add_argument("--stochastic", action="store_true")     # 탐색 노이즈 켜고 보기
p.add_argument("--seed", type=int, default=None,
               help="에피소드 시드 고정. 정책끼리 같은 목표 순서로 비교할 때 사용")
a = p.parse_args()


def available(d):
    return sorted(int(re.search(r"_(\d+)_steps", f).group(1))
                  for f in glob.glob(f"{d}/reach_*_steps.zip"))


if a.list:
    ck = available(a.out)
    print(f"{a.out} 체크포인트: {ck if ck else '없음'}")
    if os.path.exists(f"{a.out}/ppo_reach.zip"):
        print(f"{a.out} 최종 정책: ppo_reach.zip")
    sys.exit(0)

# ---- 정책 + 관측 정규화 통계 로드 (둘은 반드시 짝을 맞춰야 함) ----
if a.ckpt is not None:
    zip_path = f"{a.out}/reach_{a.ckpt}_steps.zip"
    pkl_path = f"{a.out}/reach_vecnormalize_{a.ckpt}_steps.pkl"
    if not os.path.exists(zip_path):
        ck = available(a.out)
        sys.exit(f"체크포인트 없음: {zip_path}\n사용 가능: {ck if ck else '없음 (--list 로 확인)'}")
    label = f"{a.ckpt//1000}k steps"
else:
    zip_path, pkl_path = f"{a.out}/ppo_reach", f"{a.out}/vecnorm.pkl"
    label = "final"

vn = VecNormalize.load(pkl_path, DummyVecEnv([lambda: Rby1ReachEnv()]))
vn.training = False
model = PPO.load(zip_path, device="cpu")
print(f"loaded [{label}] from {a.out}")

env = Rby1ReachEnv()
dt = env.frame_skip * env.model.opt.timestep / a.speed   # 0.02s @ 50Hz
ep = 0
obs, _ = env.reset(seed=a.seed)
succ = []

with mujoco.viewer.launch_passive(env.model, env.data) as v:
    next_t = time.time()
    while v.is_running():
        act, _ = model.predict(vn.normalize_obs(obs), deterministic=not a.stochastic)
        obs, r, term, trunc, info = env.step(act)
        v.sync()
        if trunc:                                        # 에피소드 끝 -> 새 목표
            succ.append(info["is_success"])
            print(f"[{label}] ep {ep:3d}  dist {info['dist']*1000:5.1f}mm  "
                  f"{'SUCCESS' if info['is_success'] else 'fail   '}  "
                  f"(누적 성공률 {np.mean(succ)*100:3.0f}%, n={len(succ)})")
            ep += 1
            # 시드 고정 시 정책이 달라도 목표 순서가 동일해짐
            obs, _ = env.reset(seed=None if a.seed is None else a.seed + ep)
            time.sleep(0.4)                              # 목표 바뀌는 순간 눈으로 보이게
            next_t = time.time()
        next_t += dt
        time.sleep(max(0.0, next_t - time.time()))
