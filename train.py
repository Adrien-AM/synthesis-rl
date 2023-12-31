import gymnasium as gym
from stable_baselines3 import PPO, DQN

env = gym.make("LunarLander-v2")
observation, info = env.reset()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e7, progress_bar=True)
vec_env = model.get_env()
obs = vec_env.reset()

model.save("./model.pt")
env.close()