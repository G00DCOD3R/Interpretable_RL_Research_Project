import gymnasium as gym
import numpy as np
import random
from stable_baselines3 import TD3
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.metrics import mean_squared_error, mean_absolute_error

import copy 
import graphviz 
import matplotlib.pyplot as plt 
import time 

import pickle 


"""
Check MSE of both policies on the training dataset 
Also generate pdfs of both trees 
"""
def main(): 

    expert = TD3.load("InvertedPendulum_models/td3_invpendulum-100k")

    with open('InvPen_Trees/debug_good.dump', 'rb') as file:
        good = pickle.load(file) 
    with open('InvPen_Trees/debug_bad.dump', 'rb') as file:
        bad = pickle.load(file) 
    with open('InvPen_Trees/test_dataset.dump', 'rb') as file:
        dataset = pickle.load(file) 

    # dataset = dataset[:-5000]
    X_states, y_actions = zip(*dataset)
    X_states = np.array(list(X_states)) 
    y_actions = np.array(list(y_actions)) 


    y_good = good.predict(X_states)
    y_bad = bad.predict(X_states)

    print("good mse = {}".format(mean_squared_error(y_good, y_actions)))
    print("bad mse = {}".format(mean_squared_error(y_bad, y_actions)))


    # print("good mse = {}".format(mean_absolute_error(y_good, y_actions)))
    # print("bad mse = {}".format(mean_absolute_error(y_bad, y_actions)))

    # dt = DecisionTreeRegressor(max_leaf_nodes = 15)
    # dt.fit(X_states, y_actions) 

    # y_real = dt.predict(X_states) 
    # print("real mse = {}".format(mean_squared_error(y_real, y_actions)))

    dot_data = export_graphviz(good, out_file=None, feature_names=['cart pos', 'angle', 'velocity', 'ang_vel'], class_names=['force'], filled = True)

    render_tree(dot_data, "good_tree")

    dot_data = export_graphviz(bad, out_file=None, feature_names=['cart pos', 'angle', 'velocity', 'ang_vel'], class_names=['force'], filled = True)

    render_tree(dot_data, "bad_tree")

    while True: 
        validate(bad, seeds = [random.randint(1, 10 ** 16)], visualisation=False, good = good, expert = expert)
        time.sleep(1)

def render_tree(dot_data, name):
    graph = graphviz.Source(dot_data)
    graph.render(name) 
    # graph.view()

def validate(policy, seeds = range(10, 20), visualisation = False, good = None, expert = None): 
    if visualisation:
        validation_env = gym.make('InvertedPendulum-v5', render_mode = "human")
    else:
        validation_env = gym.make('InvertedPendulum-v5', reset_noise_scale = 0.01, max_episode_steps = 10000)
        # validation_env = gym.make('Swimmer-v5')
    
    # For each seed run a simulation 
    results = [] 
    for seed in seeds:

        cur_result = 0 
        obs, _states = validation_env.reset(seed = seed)
        last_obs = obs 

        while True:
            action = policy.predict(obs.reshape(1, -1)).flatten()

            obs, rewards, terminated, truncated, info = validation_env.step(action)
            cur_result += rewards 

            # print what was the last state, last action, good policy (decision tree) action and expert action 
            if terminated or truncated:
                # validation_env.close()
                if good is not None and expert is not None:
                    print("terminating") 
                    print(last_obs) 
                    print(action)
                    print("expert got {} \t good_dt got {}\n".format(expert.predict(obs), good.predict(obs.reshape(1, -1))))
                break  

            last_obs = obs 
        
        results.append(cur_result) 

    results = np.array(results) 
    # result /= len(seeds) 
    validation_env.close() 
    return results.mean(), results.std(), np.median(results)


if __name__ == "__main__":
    main() 