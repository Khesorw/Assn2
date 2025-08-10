# ppo_tuned.py
import matplotlib
matplotlib.use("Agg")   # no GUI

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import create3_env  


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        
        r = self.locals.get("rewards")
        d = self.locals.get("dones")
        if r is not None:
            self.episode_reward += float(r[0])
        if d is not None and bool(d[0]):
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0
        return True


def make_env():
    env = gym.make("create3_env/CreateRedBall-v0")   
    env = Monitor(env)  # record episode returns & length
    return env

env = DummyVecEnv([make_env])  # single env vectorized

# ------------------------
policy_kwargs = dict(net_arch=[64, 64])  # modest network

# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=1e-4,    
#     n_steps=100,          
#     batch_size=50,         
#     n_epochs=10,           
#     gamma=0.98,
#     gae_lambda=0.95,
#     clip_range=0.2,
#     ent_coef=0.02,         
#     vf_coef=0.25,          
#     policy_kwargs=policy_kwargs,
#     verbose=1,
#     seed=2001,
# )

#tuned ppo

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,  # Increased from 1e-4 for faster learning
    n_steps=2048,        # Increased from 100 for more experience per update
    batch_size=64,       # Increased from 50 for more stable gradients
    n_epochs=10,         # Keep the same
    gamma=0.99,          # Increased from 0.98 for longer-term planning
    gae_lambda=0.95,     # Keep the same
    clip_range=0.2,      # Keep the same initially
    ent_coef=0.01,       # Reduced from 0.02 for less exploration noise
    vf_coef=0.5,         # Increased from 0.25 for better value function learning
    max_grad_norm=0.5,   # Add gradient clipping for stability
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=2001,
)

#train
callback = RewardLoggerCallback()
total_timesteps = 20000   # around 200 episodes 
model.learn(total_timesteps=total_timesteps, callback=callback)


rewards = np.array(callback.episode_rewards)
np.save("ppo_tuned_episode_rewards.npy", rewards)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(rewards) + 1), rewards, marker='o', markersize=3, linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO (tuned) Training Rewards per Episode")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("ppo_tuned_rewards_plot.png", dpi=300)
plt.close()

print("Saved!")



















