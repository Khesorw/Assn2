import gymnasium as gym
import create3_env  

# Create env
env = gym.make("create3_env/CreateRedBall-v0", render_mode="human")
print('Environment created successfully')

# Reset
observation, info = env.reset()
print(f"Initial observation: {observation}")


for _ in range(10):
    action = env.action_space.sample()  # pick a random action
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step -> obs={observation}, reward={reward}, terminated={terminated}, truncated={truncated}")

    if terminated or truncated:
        observation, info = env.reset()
        print(f"Episode reset: {observation}")

env.close()
print("Environment closed")
