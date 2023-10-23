#!/usr/bin/env/python
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure

# Sets up logger (ref notes in drive for link)
model_path = "./highway_ppo_merge/" #just update this between models/scenarios
log_path = model_path + "log/"
save_path = model_path + "model/"
logger = configure(log_path, ["stdout", "csv", "tensorboard"])

env = gym.make("merge-v0", render_mode='rgb_array')
# model = DQN('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[256, 256]),
#               learning_rate=5e-4,
#               buffer_size=15000,
#               learning_starts=200,
#               batch_size=32,
#               gamma=0.8,
#               train_freq=1,
#               gradient_steps=1,
#               target_update_interval=50,
#               verbose=1,
#               stats_window_size=100,
#               tensorboard_log="highway_ppo[0]/")
model = PPO('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              batch_size=32,
              gamma=0.8,
              verbose=1,
              stats_window_size=100,
              tensorboard_log=model_path)
model.set_logger(logger)  #sets new logger, overwriting tensorboard_log config
model.learn(int(2e4))
model.save(save_path)

# Load and test saved model
model = PPO.load(save_path)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()