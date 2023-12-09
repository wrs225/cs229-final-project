from sklearn import tree
from sympy import true
import utils_highway as utils
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from multiprocess import Pool

#Possible ways to improve:
## PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=your_value)
# training_data_X_pca = pca.fit_transform(training_data_X)
# test_data_X_pca = pca.transform(test_data_X)
# clf = tree.DecisionTreeClassifier(class_weight="balanced", ccp_alpha=your_value)
# clf.fit(training_data_X_pca, training_data_Y)

## ICA
# from sklearn.decomposition import FastICA
# ica = FastICA(n_components=your_value)
# training_data_X_ica = ica.fit_transform(training_data_X)
# test_data_X_ica = ica.transform(test_data_X)
# clf = tree.DecisionTreeClassifier(class_weight="balanced", ccp_alpha=your_value)
# clf.fit(training_data_X_ica, training_data_Y)

## FEATURE SELECTION
# from sklearn.feature_selection import SelectKBest, f_classif
# selector = SelectKBest(f_classif, k=your_value)
# training_data_X_selected = selector.fit_transform(training_data_X, training_data_Y)
# test_data_X_selected = selector.transform(test_data_X)
# clf = tree.DecisionTreeClassifier(class_weight="balanced", ccp_alpha=your_value)
# clf.fit(training_data_X_selected, training_data_Y)

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

if __name__ == "__main__":
   NUM_THREADS = 6 #Change this according to how many threads you can spare
   training_data_X, training_data_Y = utils.read_data_csv('data_merge_v0','data_merge.csv')
   test_data_X, test_data_Y = utils.read_data_csv('data_merge_v0','data_merge_test.csv')
   clf = tree.DecisionTreeClassifier(class_weight="balanced",ccp_alpha=0.0002)
   results = utils.parallelized_data_sweep(clf, "basic_decision_tree", training_data_X, training_data_Y, NUM_THREADS, starting_datas = 10)
   plt.plot(results)