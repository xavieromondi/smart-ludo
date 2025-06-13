# train_agent.py
from stable_baselines3 import PPO
from ludo_env import LudoEnv

env = LudoEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ludo_ai_model")
