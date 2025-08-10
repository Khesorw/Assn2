import gymnasium as gym
import create3_env  

env = gym.make("create3_env/CreateRedBall-v0")
print('Environment created successfully')

obs, info = env.reset()
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Reward: {reward}, Detected: {info['red_ball_detected']}, Obs shape: {obs.shape}")
env.close()