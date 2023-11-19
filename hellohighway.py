#!/usr/bin/env/python
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.logger import configure
import time
import csv

# TODO: Create a dictionary of models and scenarios, and loop through them
#   Use f"{model_dir}/{scenario_dir}" to create paths
#   https://www.youtube.com/watch?v=dLP-2Y6yu70&ab_channel=sentdex, from 1:33
# Initializes dictionary of all models and scenarios
model_type = ["dqn", "ppo"]
scenario_type = ["highway-fast-v0", 'merge-v0', 'roundabout-v0', 'intersection-v0',
            'roundabout-v0']


# Creates csv file for logged data
def create_csv(path, data):
    with open(path+".csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "mean_reward"])
        
        for row in data:
            writer.writerow(row)
            
    return path


# Sets up save/data paths
def set_path(model_name, scenario_name):
    model_path = f"./{model_name}/{scenario_name}/"
    log_path = model_path + "log/"
    save_path = model_path + "model/"
    return model_path, log_path, save_path


def create_environment(scenario_name):
    return gym.make(scenario_name, render_mode='rgb_array')


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
    model.save(save_path)
    print("------- %.2f seconds to train -------" % (time.time() - start_time))
    
    return model, save_path, log_path


# Loads and tests saved model
def test_model(model_name, scenario_name, save_path, log_path, episodes=100):
    if model_name == "dqn":
        model = DQN.load(save_path)
    elif model_name == "ppo":
        model = PPO.load(save_path)
    else:
        raise Exception("Invalid model name")
    
    env = gym.make(scenario_name, render_mode='rgb_array')
    
    sum_reward = 0.0
    mean_reward = 0.0
    reward_data = []
    
    for ep in range(episodes):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
        
        sum_reward += reward
        mean_reward = sum_reward/(ep+1)
        # print("ep=", ep, "\tep_reward=", reward, "\t mean_reward=", mean_reward)
        reward_data.append([ep, reward, mean_reward])

    env.close()
    
    csv_save_path = create_csv(log_path, reward_data)
    
    return csv_save_path


def main():
    # To run specific models/scenarios, uncomment the following lines 
    # model_trained, save_path, log_path = train_model(model_type[0], scenario_type[0])
    # test_model(model_type[0], scenario_type[0], save_path, log_path, episodes=100)
    # print("complete!")
    
    # To run all the models and scenarios, uncomment the following lines
    start_all_time = time.time()
    for model in model_type:
        for scenario in scenario_type:
            model_trained, save_path, log_path = train_model(model, scenario)
            test_model(model, scenario, save_path, log_path, episodes=100)
            print("completed ", model, " on ", scenario)
    print("------- %.2f seconds to train and test all -------" % (time.time() - start_all_time))
    
    return 0


if __name__ == "__main__":
    main()