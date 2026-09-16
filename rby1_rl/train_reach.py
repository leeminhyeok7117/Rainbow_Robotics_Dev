import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from rby1_reach_env import Rby1ReachEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--out", default="runs/reach")
    a = p.parse_args()

    venv = SubprocVecEnv([lambda: Rby1ReachEnv() for _ in range(a.n_envs)])
    venv = VecMonitor(venv, info_keywords=("is_success",))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO("MlpPolicy", venv, verbose=1, device="cpu",
                n_steps=512, batch_size=1024, n_epochs=10,
                gamma=0.98, gae_lambda=0.95, learning_rate=3e-4,
                ent_coef=0.0, policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=a.out)
    model.learn(total_timesteps=a.steps)
    os.makedirs(a.out, exist_ok=True)
    model.save(f"{a.out}/ppo_reach"); venv.save(f"{a.out}/vecnorm.pkl")
    print("saved to", a.out)

if __name__ == "__main__":
    main()
