import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from rby1_pick_env import Rby1PickEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--every", type=int, default=100_000)
    p.add_argument("--out", default="runs/pick")
    p.add_argument("--reward-mode", choices=["sparse", "dense"], default="dense")
    a = p.parse_args()
    mode = a.reward_mode
    print(f"reward_mode = {mode}")

    venv = SubprocVecEnv([(lambda m=mode: Rby1PickEnv(reward_mode=m)) for _ in range(a.n_envs)])
    venv = VecMonitor(venv, info_keywords=("is_success",))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    cb = CheckpointCallback(save_freq=max(a.every // a.n_envs, 1), save_path=a.out,
                            name_prefix="pick", save_vecnormalize=True, verbose=0)
    model = PPO("MlpPolicy", venv, verbose=1, device="cpu",
                n_steps=1024, batch_size=1024, n_epochs=10,
                gamma=0.98, gae_lambda=0.95, learning_rate=3e-4,
                ent_coef=0.005, policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=a.out)
    model.learn(total_timesteps=a.steps, callback=cb)
    model.save(f"{a.out}/ppo_pick")
    venv.save(f"{a.out}/vecnorm.pkl")
    print("saved to", a.out)

if __name__ == "__main__":
    main()
