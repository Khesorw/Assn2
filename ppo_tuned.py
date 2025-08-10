# ppo_improved.py
import matplotlib
matplotlib.use("Agg")  # no GUI
import create3_env  
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# -----------------------
# Callback to log episode rewards
# -----------------------
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

# -----------------------
# Make environment
# -----------------------
def make_env():
    env = gym.make("create3_env/CreateRedBall-v0")
    env = Monitor(env)
    return env

venv = DummyVecEnv([make_env])

# Use VecNormalize to normalize observations and (optionally) rewards
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

# -----------------------
# Hyperparameters (tuned)
# -----------------------
initial_lr = 5e-5

def lr_schedule(progress_remaining):
    # progress_remaining goes from 1 -> 0
    return initial_lr * progress_remaining

policy_kwargs = dict(net_arch=[64, 64])

model = PPO(
    "MlpPolicy",
    venv,
    learning_rate=lr_schedule,
    n_steps=100,            # match episode length
    batch_size=50,          # <= n_steps
    n_epochs=8,
    gamma=0.98,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,          # stronger entropy for exploration
    vf_coef=0.1,            # reduce critic weight
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=42,
)

# Optionally checkpoint model during training
checkpoint_cb = CheckpointCallback(save_freq=2000, save_path="./checkpoints/", name_prefix="ppo_model")

# logger callback
reward_cb = RewardLoggerCallback()

# -----------------------
# Train
# -----------------------
total_timesteps = 20000  # ~200 episodes (20000/100)
model.learn(total_timesteps=total_timesteps, callback=[reward_cb, checkpoint_cb])

# -----------------------
# Save rewards and plot (no display)
# -----------------------
rewards = np.array(reward_cb.episode_rewards)
np.save("ppo_improved_episode_rewards.npy", rewards)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, len(rewards) + 1), rewards, marker='o', markersize=3, linewidth=1)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO (improved) Training Rewards per Episode")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("ppo_improved_rewards_plot.png", dpi=300)
plt.close()

print("Saved: ppo_improved_episode_rewards.npy and ppo_improved_rewards_plot.png")
