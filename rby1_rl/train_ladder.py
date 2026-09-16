import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from rby1_reach_env import Rby1ReachEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--every", type=int, default=25_000)
    p.add_argument("--out", default="runs/ladder")
    p.add_argument("--torque", action="store_true", help="position 대신 motor(토크) 제어")
    a = p.parse_args()
    mode = "torque" if a.torque else "position"
    print(f"control_mode = {mode}")

    # 기본인자로 묶어야 서브프로세스로 넘어갈 때 mode 가 유지된다
    venv = SubprocVecEnv([(lambda m=mode: Rby1ReachEnv(control_mode=m)) for _ in range(a.n_envs)])
    venv = VecMonitor(venv, info_keywords=("is_success",))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    cb = CheckpointCallback(save_freq=max(a.every // a.n_envs, 1), save_path=a.out,
                            name_prefix="reach", save_vecnormalize=True, verbose=0)
    model = PPO("MlpPolicy", venv, verbose=1, device="cpu",
                n_steps=512, batch_size=1024, n_epochs=10,
                gamma=0.98, gae_lambda=0.95, learning_rate=3e-4,
                ent_coef=0.0, policy_kwargs=dict(net_arch=[256, 256]))
    model.learn(total_timesteps=a.steps, callback=cb)
    print("checkpoints in", a.out)

if __name__ == "__main__":
    main()
