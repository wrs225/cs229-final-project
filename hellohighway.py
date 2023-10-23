#!/usr/bin/env/python
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

# Sets up logger
data_path = "./highway_dqn_merge[0]/log/"
logger = configure(data_path, ["stdout", "csv", "tensorboard"])

env = gym.make("merge-v0", render_mode='rgb_array')
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              stats_window_size=100,
              tensorboard_log="highway_dqn_merge[0]/")
model.set_logger(logger)  #sets new logger, overwriting tensorboard_log config
model.learn(int(2e4))
model.save("highway_dqn_merge[0]/model")

# Load and test saved model
model = DQN.load("highway_dqn_merge[0]/model")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()