from pynput import keyboard
import json
import itertools
import os
import numpy as np


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
    return training_data_X, training_data_Y

def read_data_csv(folder_name, file_name):
    data = np.loadtxt(os.path.join(os.path.join(os.getcwd(), folder_name),file_name))
    print("loaded {} datas!".format(data[:,:-1].shape[0])) 
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
            x = training_data_X.append(list(itertools.chain.from_iterable(x_input))) 
            y = training_data_Y.append(simulation[sim_tick_num]['input'])
            num_examples += 1
    
    print(np.array(training_data_X).shape)
    print(np.array(training_data_Y)[:,None].shape)
    np.savetxt(os.path.join(os.path.join(os.getcwd(), folder_name),file_name),
               np.column_stack( (np.array(training_data_X), np.array(training_data_Y)) ) )

if __name__ == "__main__":
    #print('here')
    #convert_json_to_csv('data_highway_fast_v0','data_highway_fast.csv')
    #extract_test_data('data_highway_fast_v0','data_highway_fast.csv','data_highway_test.csv')
    pass