import gymnasium as gym
import highway_env
from pynput import keyboard
import json
from datetime import datetime
from sklearn import tree
import os
import itertools
import graphviz
from sklearn import svm

now = datetime.now()


reward_coef = 0.7

file_arr = []
for filename in os.listdir(os.getcwd()):
  if 'training_data' in filename:
    print(filename)
    file = open(filename)
    file_string = file.read()
    if(file_string == None):
      print("WARN: Json File {} has nothing in it".format(filename))
      pass

    file.close()
    data = json.loads(file_string)

    file_arr.append(data)


training_data_X = []
training_data_Y = []
for file_dict in file_arr:
  for simulation_num, simulation in file_dict.items():
    for sim_tick_num in range(1, len(simulation)):
      if(simulation[-1]['reward'] > reward_coef):
        x = training_data_X.append(list(itertools.chain.from_iterable(simulation[sim_tick_num - 1]['obs'])))
        y = training_data_Y.append(simulation[sim_tick_num]['input'])
      

clf = svm.SVC()

clf.fit(training_data_X, training_data_Y)
env = gym.make("highway-fast-v0", render_mode='rgb_array')

env.configure({
  "action":{"type":"DiscreteMetaAction"},
  "simulation_frequency": 20
})


while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action = clf.predict([list(itertools.chain.from_iterable(obs))])
    obs, reward, done, truncated, info = env.step(action)

    env.render()
  print(reward)
