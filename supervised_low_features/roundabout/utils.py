from pynput import keyboard
import json
import itertools
import os
import numpy as np
import gymnasium as gym
import highway_env
import math
import csv
from multiprocess import Pool
import tqdm



def read_data_json(folder_name, reward_coef = 0.7):
    

    file_arr = []
    for filename in os.listdir(os.path.join(os.getcwd(), folder_name)):
      if 'training_data' in filename:
        print(filename)
        file = open(os.path.join(os.path.join(os.getcwd(), folder_name),filename))
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
    
    print(training_data_X.shape())

    return training_data_X, training_data_Y

def read_data_csv(folder_name, file_name):
    data = np.loadtxt(os.path.join(os.path.join(os.getcwd(), folder_name),file_name))
    print("loaded {} datas with {} features!".format(data[:,:-1].shape[0],data[:,:-1].shape[1])) 
    return data[:,:-1], data[:,-1]

def extract_test_data(folder_name, file_name,newfile_name, points = 10000):
    print("are you sure you want to run this? You can ruin the data(press any key besides ctrl-c to continue)")
    input()
    data = np.loadtxt(os.path.join(os.path.join(os.getcwd(), folder_name),file_name))
    print(data[-points:,:].shape)
    print(os.path.join(os.path.join(os.getcwd(), folder_name),newfile_name))
    np.savetxt(os.path.join(os.path.join(os.getcwd(), folder_name),newfile_name), data[-points:,:])
    print(data[:points,:])
    print(os.path.join(os.path.join(os.getcwd(), folder_name),file_name))
    np.savetxt(os.path.join(os.path.join(os.getcwd(), folder_name),file_name), data[:-points,:])

    
def convert_json_to_csv(folder_name, file_name, reward_coef = 0.7):
    

    file_arr = []
    for filename in os.listdir(os.path.join(os.getcwd(), folder_name)):
      if 'training_data' in filename:
        print(filename)
        file = open(os.path.join(os.path.join(os.getcwd(), folder_name),filename))
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
            x = training_data_X.append(x_input) 
            y = training_data_Y.append(simulation[sim_tick_num]['input'])
            num_examples += 1
    
    print(np.array(training_data_X).shape)
    print(np.array(training_data_Y)[:,None].shape)
    np.savetxt(os.path.join(os.path.join(os.getcwd(), folder_name),file_name),
               np.column_stack( (np.array(training_data_X), np.array(training_data_Y)) ) )
    
def parallelized_data_sweep(clf, name, training_data_X, training_data_Y, test_data_X, test_data_Y, num_threads,starting_datas=4):
    train_example_sweep_iterator = 0
    iterations = math.trunc(math.log2(len(training_data_X)))
    print("sweeping exponentially 2^n up to n={}".format(iterations))

    file = open('{}_data.csv'.format(name), 'w', newline='')
    writer = csv.writer(file,delimiter=' ', quotechar='|')
    writer.writerow(['data_points','training_accuracy','simulation_reward', 'test_accuracy'])
    file.close()

    output_data_points = np.array(range(starting_datas,iterations+1))
    output_reward = np.zeros(len(range(starting_datas,iterations+1)))
    if(starting_datas != -1):
      for i in range(starting_datas,iterations + 1):
          print("Training {} on {} examples!".format(name, 2**i))
          #clf = make_pipeline(StandardScaler(),svm.SVC(class_weight='balanced', cache_size=1000))
          clf.fit(training_data_X[0:2**i], training_data_Y[0:2**i])

          train_accuracy = clf.score(training_data_X[0:2**i],training_data_Y[0:2**i])
          print("{} trained with accuracy {} on training set".format(name, train_accuracy))

          test_accuracy = clf.score(test_data_X, test_data_Y)
          print("resulting test accuracy is:".format(test_accuracy))


          NUM_EPOCHS = 1000

          def parallelized_simulaton(e):
          
              env = gym.make("roundabout-v0", config= {
                "action":{"type":"DiscreteMetaAction"},
                "simulation_frequency": 10
              })
              """,render_mode='rgb_array'"""

              epochs = 0
              reward_sum = 0
              NUM_EPOCHS = 5
              while epochs < NUM_EPOCHS:
                  done = truncated = False
                  obs, info = env.reset()
                  while not (done or truncated):
                      inner = [list(itertools.chain.from_iterable(obs))]
                      

                      
                      action = clf.predict(inner)
                      
                      obs, reward, done, truncated, info = env.step(action)

                  #env.render()
                  reward_sum += reward
                  epochs += 1

              return reward_sum

          with Pool(num_threads) as p:
              clf_list = [clf] * (NUM_EPOCHS // 5)
              ep = range(NUM_EPOCHS // 5)

              reward_sum = sum(list(tqdm.tqdm(p.imap_unordered(parallelized_simulaton,ep), total=len(ep))))
          
          output_reward[i - starting_datas] = reward_sum/NUM_EPOCHS
          file = open('{}_data.csv'.format(name), 'a', newline='')
          writer = csv.writer(file,delimiter=' ', quotechar='|')
          writer.writerow([2**i,train_accuracy,output_reward[i - starting_datas],test_accuracy])
          file.close()

          print("{} trained with accuracy {} on training set".format(name, train_accuracy))
          print("testing competed with an average reward of {} over {} simulations".format(reward_sum / NUM_EPOCHS, NUM_EPOCHS))
    else:
        print("Training {} on {} examples!".format(name, training_data_X.shape[0]))
        #clf = make_pipeline(StandardScaler(),svm.SVC(class_weight='balanced', cache_size=1000))
        clf.fit(training_data_X, training_data_Y)
        train_accuracy = clf.score(training_data_X,training_data_Y)
        print("{} trained with accuracy {} on training set".format(name, train_accuracy))
        NUM_EPOCHS = 1000
        def parallelized_simulaton(e):
        
            env = gym.make("roundabout-v0", config= {
              "action":{"type":"DiscreteMetaAction"},
              "simulation_frequency": 10
            },render_mode='rgb_array')
            """,render_mode='rgb_array'"""
            epochs = 0
            reward_sum = 0
            NUM_EPOCHS = 5
            while epochs < NUM_EPOCHS:
                done = truncated = False
                obs, info = env.reset()
                while not (done or truncated):
                    inner = [list(itertools.chain.from_iterable(obs))]
                    
                    
                    action = clf.predict(inner)
                    
                    obs, reward, done, truncated, info = env.step(action)
                    env.render()
                reward_sum += reward
                epochs += 1
            return reward_sum
        if(num_threads != 1):
          with Pool(num_threads) as p:
              clf_list = [clf] * (NUM_EPOCHS // 5)
              ep = range(NUM_EPOCHS // 5)
              reward_sum = sum(list(tqdm.tqdm(p.imap_unordered(parallelized_simulaton,ep), total=len(ep))))
        else:
          print('running normal simulation')
          reward_sum = 0
          for i in range(NUM_EPOCHS):
            reward_sum += parallelized_simulaton(0)
        
        #output_reward[i - starting_datas] = reward_sum/NUM_EPOCHS
        #file = open('{}_data.csv'.format(name), 'a', newline='')
        #writer = csv.writer(file,delimiter=' ', quotechar='|')
        #writer.writerow([training_data_Y.shape[0],train_accuracy,output_reward[i - starting_datas]])
        file.close()

        print("{} trained with accuracy {} on training set".format(name, train_accuracy))
        print("testing competed with an average reward of {} over {} simulations".format(reward_sum / NUM_EPOCHS, NUM_EPOCHS))

    return output_data_points, output_reward


if __name__ == "__main__":
    #read_data_json('data_highway_fast_v0')
    #print('here')
    convert_json_to_csv('data_roundabout_v0','data_roundabout_train_2.csv')
    #extract_test_data('data_roundabout_v0','data_roundabout_train.csv','data_roundabout_test.csv', points = 1000)
    #read_data_csv('data_highway_fast_v0','data_highway_fast_2.csv')
    pass