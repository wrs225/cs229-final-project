#!/usr/bin/env/python
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
import time
import csv
import numpy as np


# Initializes dictionary of all models and scenarios
model_type = ["dqn", "ppo"]
scenario_type = ["highway-fast-v0", 'merge-v0', 'roundabout-v0', 'intersection-v0',
            'roundabout-v0']


# Creates csv file for logged data
def create_csv(path, name, data):
    with open(path+name+".csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "mean_reward"])
        
        for row in data:
            writer.writerow(row)
            
    return path


# Reads csv file of logged data
def read_csv(path):
    data = []
    
    with open(path+".csv", 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
            
    return data


# Sets up save/data paths
def set_path(model_name, scenario_name):
    model_path = f"./{model_name}/{scenario_name}/"
    log_path = model_path + "log/"
    save_path = model_path + "model/"
    return model_path, log_path, save_path


def create_environment(scenario_name):
    env = gym.make(scenario_name, render_mode='rgb_array')
    env.configure({
        "action":{"type":"DiscreteMetaAction"},
        "simulation_frequency": 20
    })
    # env.configure({
    #     "observation":{
    #         "type":"OccupancyGrid",
    #         "features": ["presence", "vx", "vy",]},
    #     "action":{"type":"DiscreteMetaAction"},
    #     "simulation_frequency": 20
    # })
    # env.configure({
    #     "observation": {
    #         "type": "OccupancyGrid",
    #         "vehicles_count": 15,
    #         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    #         "features_range": {
    #             "x": [-100, 100],
    #             "y": [-100, 100],
    #             "vx": [-20, 20],
    #             "vy": [-20, 20]
    #         },
    #         "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
    #         "grid_step": [5, 5],
    #         "absolute": False
    #     }
    # })
    
    return env


# Sets up the model with selected Gym environment
def select_model(model_name, scenario_name):
    env = create_environment(scenario_name)

    if model_name == "dqn":
        model = DQN('MlpPolicy', env,
			policy_kwargs=dict(net_arch=[256, 256]),
			learning_rate=5e-4,
			buffer_size=15000,
			learning_starts=200,
			batch_size=32,
			gamma=0.8,
			train_freq=1,
			gradient_steps=1,
			target_update_interval=50,
			verbose=1,
			stats_window_size=100)
        return model
    elif model_name == "ppo":
        model = PPO('MlpPolicy', env,
			policy_kwargs=dict(net_arch=[256, 256]),
			learning_rate=5e-4,
			batch_size=32,
			gamma=0.8,
			verbose=1,
			stats_window_size=100)
        return model
    else:
        raise Exception("Invalid model name")


# Trains the model while logging data  
def train_model(model_name, scenario_name):
    model = select_model(model_name, scenario_name)
    model_path, log_path, save_path = set_path(model_name, scenario_name)
    
    # logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    logger = configure(log_path, ["csv", "tensorboard"])
    model.set_logger(logger)
    
    start_time = time.time()
    print("Starting training for %s in %s..." % (model_name, scenario_name))
    model.learn(int(2e4))
    print("------- %.2f seconds to train -------" % (time.time() - start_time))
    
    model.save(save_path)
    
    return model, save_path, log_path


# Loads and tests saved model
def test_model(model_name, scenario_name, save_path, log_path, episodes=200, render=False, test_csv=None):
    if model_name == "dqn":
        model = DQN.load(save_path)
    elif model_name == "ppo":
        model = PPO.load(save_path)
    else:
        raise Exception("Invalid model name")
    
    env = create_environment(scenario_name)
    
    sum_reward = 0.0
    mean_reward = 0.0
    reward_data = []
    
    if test_csv:
        test_data = read_csv(log_path+test_csv)
        # print("test_data=", test_data, "\tlen=", len(test_data), "datatype=", type(test_data))
        ep = 0

        for data in test_data:
            done = truncated = False
            obs, info = env.reset()
            obs = np.fromstring(data[0], dtype=float, sep=' ')
            obs = np.array(obs[1:26]).reshape(5, 5)
            # print("ep=", ep, "\tobs_init=", obs, "\tlen=", len(obs), "datatype=", type(obs))
                        
            while not (done or truncated):
                action, states = model.predict(obs, deterministic=True)

                # print("ep=", ep, "\taction=", action)
                # time.sleep(1)

                obs, reward, done, truncated, info = env.step(action)

                # print("ep=", ep, "\tobs=", obs)
                # time.sleep(1)

                if render:
                    env.render()
            
            sum_reward += reward
            mean_reward = sum_reward/(ep+1)
            print("ep=", ep, "\tep_reward=", reward, "\t mean_reward=", mean_reward)
            reward_data.append([ep, reward, mean_reward])

            ep += 1

            if ep > episodes:
                break

    else:
        for ep in range(episodes):
            done = truncated = False
            obs, info = env.reset()

            while not (done or truncated):
                
                # TODO: Use test data to evaluate model
                # 0) Load test data from JSON file
                # 1) Create numpy array using test data for each input
                # 2) Predict action using model.predict()
                # 3) Use action to step through environment
                # 4) Repeat until done or truncated
                action, states = model.predict(obs, deterministic=True)
                
                # print("ep=", ep, "\taction=", action)
                # time.sleep(1)

                obs, reward, done, truncated, info = env.step(action)
                
                # print("ep=", ep, "\tobs=", obs)
                # time.sleep(0.5)
                
                if render:
                    env.render()
            
            sum_reward += reward
            mean_reward = sum_reward/(ep+1)
            # print("ep=", ep, "\tep_reward=", reward, "\t mean_reward=", mean_reward)
            reward_data.append([ep, reward, mean_reward])

    env.close()
    
    csv_save_path = create_csv(log_path, "test_res", reward_data)
    
    return csv_save_path


def main():
    # To run specific models/scenarios, uncomment the following lines:
    # model_trained, save_path, log_path = train_model(model_type[0], scenario_type[0])   #commented out to test w/o o/w trained models
    model_path, log_path, save_path = set_path(model_type[0], scenario_type[1])         #added to test w/o o/w trained models
    # test_model(model_type[0], scenario_type[1], save_path, log_path, episodes=100, render=True) #added to test on random test data
    test_model(model_type[0], scenario_type[0], save_path, log_path, render=False, test_csv='data_merge_test')  #added to test on specific test data
    print("complete!")
    
    # To run all the models and scenarios, uncomment the following lines:
    # start_all_time = time.time()
    # for model in model_type:
    #     for scenario in scenario_type:
    #         # model_trained, save_path, log_path = train_model(model, scenario) #commented out to test w/o o/w trained models
    #         model_path, log_path, save_path = set_path(model, scenario)         #added to test w/o o/w trained models
    #         test_model(model, scenario, save_path, log_path, episodes=100)
    #         print("completed ", model, " on ", scenario)
    # print("------- %.2f seconds to train and test all -------" % (time.time() - start_all_time))
    
    return 0


if __name__ == "__main__":
    main()