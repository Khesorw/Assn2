import numpy as np
import gymnasium as gym
import create3_env  




alpha = 0.09      # learning rate
gamma = 0.95     # discount factor
epsilon = 1.0    # exploration rate
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 500

env = gym.make("create3_env/CreateRedBall-v0")


# Q-table: states x actions
q_table = np.zeros((env.observation_space.n, env.action_space.n))

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    done = False
    while not done:
        # Choose action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {ep+1}/{episodes} - Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()