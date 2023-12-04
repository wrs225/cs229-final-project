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
from multiprocess import Pool
import tqdm

NUM_THREADS = 6 #Change this according to how many threads you can spare

now = datetime.now()


training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_fast.csv')

utils.paralleized_data_sweep(tree.DecisionTreeClassifier, "basic_decision_tree", training_data_X, training_data_Y, NUM_THREADS)