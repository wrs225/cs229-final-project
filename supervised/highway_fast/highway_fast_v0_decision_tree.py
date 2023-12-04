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
import math
import csv
import utils
from multiprocessing import Pool
import tqdm

NUM_THREADS = 6 #Change this according to how many threads you can spare

now = datetime.now()


training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_fast.csv')

train_example_sweep_iterator = 0
iterations = math.trunc(math.log2(len(training_data_X)))
print("sweeping exponentially 2^n up to n={}".format(iterations))

file = open('dtree_data.csv', 'w', newline='')
writer = csv.writer(file,delimiter=' ', quotechar='|')
writer.writerow(['data_points','training_accuracy','simulation_reward'])
file.close()

for i in range(4,iterations + 1):
  print("Training svm on {} examples!".format(2**i))
  #clf = make_pipeline(StandardScaler(),svm.SVC(class_weight='balanced', cache_size=1000))
  clf = tree.DecisionTreeClassifier(class_weight='balanced')
  clf.fit(training_data_X[0:2**i], training_data_Y[0:2**i])

  train_accuracy = clf.score(training_data_X[0:2**i],training_data_Y[0:2**i])
  print("svm trained with accuracy {} on training set".format(train_accuracy))


  NUM_EPOCHS = 1000

  def parallelized_simulaton(e):

    env = gym.make("highway-fast-v0", config= {
        "observation":{"type":"OccupancyGrid",
                       "features": ["presence", "vx", "vy",]},
      "action":{"type":"DiscreteMetaAction"},
      "simulation_frequency": 5
    })

    
    epochs = 0
    reward_sum = 0
    NUM_EPOCHS = 5
    while epochs < NUM_EPOCHS:
      done = truncated = False
      obs, info = env.reset()
      while not (done or truncated):
        inner = list(itertools.chain.from_iterable(obs))
        #print(len(list(itertools.chain.from_iterable(inner))))
        action = clf.predict([list(itertools.chain.from_iterable(inner))])
        obs, reward, done, truncated, info = env.step(action)

        #env.render()
      reward_sum += reward
      epochs += 1

    return reward_sum
  
  with Pool(NUM_THREADS) as p:
    clf_list = [clf] * (NUM_EPOCHS // 5)
    ep = range(NUM_EPOCHS // 5)
    
    reward_sum = sum(list(tqdm.tqdm(p.imap_unordered(parallelized_simulaton,ep), total=len(ep))))
  
  file = open('dtree_data.csv', 'a', newline='')
  writer = csv.writer(file,delimiter=' ', quotechar='|')
  writer.writerow([2**i,train_accuracy,reward_sum/NUM_EPOCHS])
  file.close()

  print("svm trained with accuracy {} on training set".format(train_accuracy))
  print("testing competed with an average reward of {} over {} simulations".format(reward_sum / NUM_EPOCHS, NUM_EPOCHS))