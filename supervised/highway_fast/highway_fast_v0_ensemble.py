import gymnasium as gym
import highway_env
from pynput import keyboard
import json
from datetime import datetime
from sklearn import tree
import os
import itertools
import graphviz
from sklearn.ensemble import HistGradientBoostingClassifier
import utils

now = datetime.now()


training_data_X, training_data_Y = utils.read_data_json('data_highway_fast_v0')

print("Training decision tree on {} examples!".format(len(training_data_X)))

clf = HistGradientBoostingClassifier(max_iter=100,class_weight='balanced').fit(training_data_X, training_data_Y)
print(clf.score(training_data_X, training_data_Y))
env = gym.make("highway-fast-v0", render_mode='rgb_array')

env.configure({
    "observation":{"type":"OccupancyGrid",
                   "features": ["presence", "vx", "vy",]},
  "action":{"type":"DiscreteMetaAction"},
  "simulation_frequency": 5
})

epochs = 0
reward_sum = 0
NUM_EPOCHS = 100
while epochs < NUM_EPOCHS:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    inner = list(itertools.chain.from_iterable(obs))
    #print(len(list(itertools.chain.from_iterable(inner))))
    action = clf.predict([list(itertools.chain.from_iterable(inner))])
    obs, reward, done, truncated, info = env.step(action)

    env.render()
  reward_sum += reward
  epochs += 1
  print(epochs)

print("testing competed with an average reward of {} over {} simulations".format(reward_sum / NUM_EPOCHS, NUM_EPOCHS)) 
  
