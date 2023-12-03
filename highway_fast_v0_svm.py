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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import math
import csv

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
num_examples = 0
for file_dict in file_arr:
  for simulation_num, simulation in file_dict.items():
    for sim_tick_num in range(1, len(simulation)):
      if(simulation[sim_tick_num]['reward'] > reward_coef):
        #x = training_data_X.append(list(itertools.chain.from_iterable(simulation[sim_tick_num - 1]['obs']))) #old obs space
            
        x_input = list(itertools.chain.from_iterable(simulation[sim_tick_num - 1]['obs']))
        x = training_data_X.append(list(itertools.chain.from_iterable(x_input))) 
        y = training_data_Y.append(simulation[sim_tick_num]['input'])
        num_examples += 1

train_example_sweep_iterator = 0
iterations = math.trunc(math.log2(num_examples))
print("sweeping exponentially 2^n up to n={}".format(iterations))

file = open('svm_data.csv', 'w', newline='')
writer = csv.writer(file,delimiter=' ', quotechar='|')
writer.writerow(['data_points','training_accuracy','simulation_reward'])
file.close()

for i in range(4,iterations + 1):
  print("Training svm on {} examples!".format(2**i))
  #clf = make_pipeline(StandardScaler(),svm.SVC(class_weight='balanced', cache_size=1000))
  clf = svm.SVC(class_weight='balanced', cache_size=1000)
  clf.fit(training_data_X[0:2**i], training_data_Y[0:2**i])
  env = gym.make("highway-fast-v0", render_mode='rgb_array')

  env.configure({
      "observation":{"type":"OccupancyGrid",
                     "features": ["presence", "vx", "vy",]},
    "action":{"type":"DiscreteMetaAction"},
    "simulation_frequency": 5
  })
  train_accuracy = clf.score(training_data_X[0:2**i],training_data_Y[0:2**i])
  print("svm trained with accuracy {} on training set".format(train_accuracy))
  epochs = 0
  reward_sum = 0
  NUM_EPOCHS = 1000
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
  
  file = open('svm_data.csv', 'a', newline='')
  writer = csv.writer(file,delimiter=' ', quotechar='|')
  writer.writerow([2**i,train_accuracy,reward_sum/NUM_EPOCHS])
  file.close()

  print("svm trained with accuracy {} on training set".format(train_accuracy))
  print("testing competed with an average reward of {} over {} simulations".format(reward_sum / NUM_EPOCHS, NUM_EPOCHS))


