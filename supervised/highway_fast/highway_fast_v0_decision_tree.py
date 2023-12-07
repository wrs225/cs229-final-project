from sklearn import tree
import utils_highway as utils
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from multiprocess import Pool
from sklearn.model_selection import cross_val_score
import graphviz 
from sklearn.model_selection import train_test_split

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

# NUM_THREADS = 8 #Change this according to how many threads you can spare
# training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_fast.csv')
# test_data_X, test_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_test.csv')

if(False):
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
    NUM_THREADS = 8
    ccp_alphas = [0.001]
    max_depths = [None] # [5, 10] not good enough, needs further testing
    min_samples_splits = [2] # 5, 10 did not perform as well, sci-kts default is 2 and it performed the best
    min_samples_leafs = [1] # tested [2, 4], but no effect really. Sci-kit reccomends 1

    # Load the training data once outside the hyperparameter loop
    training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0', 'data_highway_fast.csv')
    
    # Load the test data once outside the hyperparameter loop
    test_data_X, test_data_Y = utils.read_data_csv('data_highway_fast_v0', 'data_highway_test.csv')

    for ccp_alpha in ccp_alphas:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_samples_leaf in min_samples_leafs:
                    # Split the data into training and testing sets
                    # train_data_X, _, train_data_Y, _ = train_test_split(training_data_X, training_data_Y, test_size=0.2, random_state=42)

                    clf = tree.DecisionTreeClassifier(
                        class_weight="balanced",
                        ccp_alpha=ccp_alpha,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf
                    )

                    # Existing code for training and testing the model remains unchanged
                    results = utils.parallelized_data_sweep(clf, "basic_decision_tree", training_data_X, training_data_Y, NUM_THREADS, starting_datas=10)
                    # plt.plot(results); plt.show()

                    # Evaluate the model on the test set
                    test_accuracy = clf.score(test_data_X, test_data_Y)
                    print(f"\nDecision Tree with parameters: ccp_alpha={ccp_alpha}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
                    print(f"Testing set accuracy: {test_accuracy}")
                    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)  
                    graph = graphviz.Source(dot_data)  
                    graph.render("decision_tree")
                
  #  for i in range(0, len(ccp_alphas)):
  #   NUM_THREADS = 8 #Change this according to how many threads you can spare
  #   training_data_X, training_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_fast.csv')
  #   test_data_X, test_data_Y = utils.read_data_csv('data_highway_fast_v0','data_highway_test.csv')
  #   clf = tree.DecisionTreeClassifier(class_weight="balanced", ccp_alpha = ccp_alphas[i]) # max_depth = , min_samples_leaf, min_samples_split
  #   results = utils.parallelized_data_sweep(clf, "basic_decision_tree", training_data_X, training_data_Y, NUM_THREADS, starting_datas = 10)
  #   #plt.plot(results); plt.show()

  #   # get each test values 
  #   train_scores.append(clf.score(training_data_X, training_data_Y))
  #   test_scores.append(clf.score(test_data_X, test_data_Y))
  #   print(f"\nNode Count at ccp_alpha = {ccp_alphas[i]} is {clf.fit(training_data_X, training_data_Y).tree_.node_count}\n")
  #   #cross_val_scores.append(cross_val_score(clf, test_data_X, test_data_Y, cv=5, scoring='accuracy').mean())
  #   cross_val_scores.append(cross_val_score(clf, training_data_X, training_data_Y, cv=5, scoring='accuracy').mean())
  #   print(f"\nCross Val Score at ccp_alpha = {ccp_alphas[i]} is {cross_val_scores[i]}\n")
  #   dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, special_characters=True)  
  #   graph = graphviz.Source(dot_data)  
  #   graph.render("decision_tree")
   
   # after exiting loop
  #  fig, ax = plt.subplots()
  #  ax.set_xlabel("alpha")
  #  ax.set_ylabel("accuracy")
  #  ax.set_title("Accuracy vs alpha for training and testing sets")
  #  ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
  #  ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
  #  ax.legend()
  #  plt.show()