import gymnasium as gym
from stable_baselines3 import PPO, DQN

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

model = DQN("MlpPolicy", env)
model.load("./model.pt")
vec_env = model.get_env()
obs = vec_env.reset()

for i in range(10000):
    action, _states = model.predict(obs)
    observation, reward, terminated, info = vec_env.step(action)
    env.render()
    
    if terminated:
        observation, info = env.reset()

env.close()