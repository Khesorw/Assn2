import numpy as np
import gymnasium as gym
import create3_env  # Your package with CreateRedBall-v0
import matplotlib.pyplot as plt

# Create environment
env = gym.make("create3_env/CreateRedBall-v0")

# Q-learning hyperparameters
alpha = 0.1          # learning rate
gamma = 0.99         # discount factor
epsilon = 0.1        # exploration rate
num_episodes = 500
max_steps_per_episode = 100

# Initialize Q-table
# Assume discrete observation and action spaces
Q = np.zeros((env.observation_space.n, env.action_space.n))

episode_rewards = []

for episode in range(num_episodes):
    observation, info = env.reset()
    total_reward = 0
    for step in range(max_steps_per_episode):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[observation])

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Q-learning update
        best_next_action = np.argmax(Q[next_observation])
        td_target = reward + gamma * Q[next_observation, best_next_action]
        td_error = td_target - Q[observation, action]
        Q[observation, action] += alpha * td_error

        observation = next_observation

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)
    if (episode+1) % 50 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()

# Plot the total reward per episode
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning on CreateRedBall-v0')
plt.show()
