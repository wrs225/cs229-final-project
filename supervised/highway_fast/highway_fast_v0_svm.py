import utils
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from multiprocess import Pool
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


NUM_THREADS = 8 #Change this according to how many threads you can spare

training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_fast.csv')
test_data_X, test_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_test.csv')


clf = make_pipeline(StandardScaler(),SVC(class_weight="balanced",cache_size = 1000))
results = utils.paralleized_data_sweep(clf, "svm_sigmoid", training_data_X, training_data_Y, NUM_THREADS, starting_datas = 10)

plt.plot(results[0],results[1])
plt.show()