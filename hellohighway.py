#!/usr/bin/env/python
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
import time

# Sets up logger (ref notes in drive for more info)
#   Update next two lines between models/scenarios
scenario = "intersection-v0"
model_path = "./highway_ppo_intersection_v0/" 

log_path = model_path + "log/"
save_path = model_path + "model/"
logger = configure(log_path, ["stdout", "csv", "tensorboard"])

env = gym.make(scenario, render_mode='rgb_array')
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
start_time = time.time()
model.learn(int(2e4))
model.save(save_path)
print("------- %.2f seconds to train -------" % (time.time() - start_time))

# Load and test saved model
model = PPO.load(save_path)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()