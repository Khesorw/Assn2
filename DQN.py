import gymnasium as gym
import create3_env  
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')  #
from stable_baselines3.common.vec_env import DummyVecEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]

        # Check if episode is done
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        return True

# Create red ball env
env = gym.make("create3_env/CreateRedBall-v0")
env = DummyVecEnv([lambda: env])

#DQN model 
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,        
    buffer_size=50000,          
    learning_starts=1000,      
    batch_size=32,              
    gamma=0.99,                
    target_update_interval=500, 
    train_freq=4,              
    verbose=1,
    seed=42,
)

# Create callback instance
callback = RewardLoggerCallback()

model.learn(total_timesteps=10000, callback=callback)

plt.plot(callback.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Rewards per Episode')
plt.savefig("dqn_rewards.png")
print("Plot saved to dqn_rewards.png")











# # Custom callback to log rewards per episode
# class RewardLoggerCallback(BaseCallback):
#     def __init__(self):
#         super().__init__()
#         self.episode_rewards = []
#         self.episode_reward = 0

#     def _on_step(self) -> bool:

#         self.episode_reward += self.locals["rewards"][0]

#         # Check if episode is done
#         if self.locals["dones"][0]:
#             self.episode_rewards.append(self.episode_reward)
#             self.episode_reward = 0
#         return True

# # Use your environment here
# env = gym.make("create3_env/CreateRedBall-v0")


# model = DQN(
#     "MlpPolicy",
#     env,
#     learning_rate=1e-3,         
#     buffer_size=50000,          
#     learning_starts=1000,       
#     batch_size=32,              
#     gamma=0.99,                 
#     target_update_interval=500, 
#     train_freq=2,               
#     verbose=1,
#     seed=42,
# )

# callback = RewardLoggerCallback()

# model.learn(total_timesteps=10000, callback=callback)

# plt.plot(callback.episode_rewards)
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.title('DQN Training Rewards per Episode')
# plt.show()