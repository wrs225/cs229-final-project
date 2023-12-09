from sklearn import tree
import supervised_low_features.highway_fast.utils as utils
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from multiprocess import Pool

from sklearn.ensemble import AdaBoostClassifier



NUM_THREADS = 8 #Change this according to how many threads you can spare

training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_fast.csv')
test_data_X, test_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_test.csv')


clf = tree.DecisionTreeClassifier(class_weight="balanced",max_depth=5)
clf = AdaBoostClassifier(clf,n_estimators=100)
results = utils.paralleized_data_sweep(clf, "adaboost_d5", training_data_X, training_data_Y, NUM_THREADS, starting_datas = 10)

plt.plot(results)