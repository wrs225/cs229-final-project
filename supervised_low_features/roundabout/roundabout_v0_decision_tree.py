import utils 
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from multiprocess import Pool
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


NUM_THREADS = 7 #Change this according to how many threads you can spare

training_data_X, training_data_Y = utils.read_data_csv('data_roundabout_v0','data_roundabout_train.csv')
test_data_X, test_data_Y = utils.read_data_csv('data_roundabout_v0','data_roundabout_test.csv')


clf = DecisionTreeClassifier(class_weight="balanced")
results = utils.parallelized_data_sweep(clf, "dtree", training_data_X, training_data_Y, test_data_X, test_data_Y, NUM_THREADS, starting_datas = 6)

plt.plot(results[0],results[1])
plt.show()