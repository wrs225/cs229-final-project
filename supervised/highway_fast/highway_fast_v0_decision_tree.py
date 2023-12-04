from sklearn import tree
import utils
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from multiprocess import Pool


NUM_THREADS = 8 #Change this according to how many threads you can spare

training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_fast.csv')
test_data_X, test_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_test.csv')


if(True):
  clf = tree.DecisionTreeClassifier(class_weight="balanced")

  path = clf.cost_complexity_pruning_path(training_data_X, training_data_Y)
  ccp_alphas, impurities = path.ccp_alphas[:-1:len(path.ccp_alphas)//15], path.impurities
  clfs = []
  for ccp_alpha in tqdm.tqdm(ccp_alphas):
      clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
      clf.fit(training_data_X, training_data_Y)
      clfs.append(clf)

  train_scores = [clf.score(training_data_X, training_data_Y) for clf in tqdm.tqdm(clfs)]
  test_scores = [clf.score(test_data_X, test_data_Y) for clf in tqdm.tqdm(clfs)]

  print('Obtained maximum test accuracy with {}'.format(ccp_alphas[np.argmax(test_scores)]))
  fig, ax = plt.subplots()
  ax.set_xlabel("alpha")
  ax.set_ylabel("accuracy")
  ax.set_title("Accuracy vs alpha for training and testing sets")
  ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
  ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
  ax.legend()
  plt.show()

clf = tree.DecisionTreeClassifier(class_weight="balanced",ccp_alpha=0.0002)
results = utils.paralleized_data_sweep(clf, "basic_decision_tree", training_data_X, training_data_Y, NUM_THREADS, starting_datas = 10)

plt.plot(results)