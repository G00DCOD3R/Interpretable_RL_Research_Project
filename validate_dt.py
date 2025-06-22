import gymnasium as gym
import numpy as np
import random
from stable_baselines3 import TD3
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz

import copy 
import graphviz 
import matplotlib.pyplot as plt 
import time 

import pickle 

def main():

    expert, good = None, None 
    means = [] 
    stds = [] 
    xs = range(5, 100, 5)

    for MAX_NODES in xs:

        with open(f'results/Hopper/{MAX_NODES}.dump', 'rb') as file:
            dt = pickle.load(file) 

        # expert = TD3.load("InvertedPendulum_models/td3_invpendulum-100k")
        # with open('InvPen_Trees/debug_good.dump', 'rb') as file:
        #     good = pickle.load(file) 


        # discrete_expert(good) 

        # return 

        mean, std, median = validate(dt, seeds = range(100))
        print(MAX_NODES, mean, std, median)

        means.append(mean)
        stds.append(std) 

        # print("validation:\nmiu = {}\tstd = {}\tmedian = {}".format(mean, std, median)) 

    means = np.array(means) 
    stds = np.array(stds) 
    nice_plot(xs, means, stds) 
    # while True:
    #     validate(dt, seeds = [random.randint(1, 1000)], visualisation=True, good = good, expert = expert)

def nice_plot(xs, means, stds): 

    plt.plot(xs, means)
    plt.fill_between(xs, means - stds, means + stds, alpha = 0.2) 
    plt.savefig("Hopper_results")
    plt.show() 

"""
Discretize an action 
"""
def disc_map(action): 

    return np.array([round(action[0], 1)])

    if action[0] < 0:
        return np.array([-3])
    else:
        return np.array([3])

"""
check how well discrete expert would perform 
"""
def discrete_expert(policy):
    # env = gym.make("InvertedPendulum-v5", render_mode = "human", reset_noise_scale = 0.1, max_episode_steps = 10000)
    env = gym.make("Hopper-v5", render_mode = "human", reset_noise_scale = 0.1, max_episode_steps = 10000)

    while True: 

        obs, _states = env.reset() 
        cur_result = 0 

        while True: 
            # action, _states = policy.predict(obs)
            action = policy.predict(obs.reshape(1, -1))

            # action = disc_map(action)

            obs, rewards, terminated, truncated, info = env.step(action)
            cur_result += rewards 

            if terminated or truncated: 
                break 
        print("result = {}".format(cur_result))


def validate(policy, seeds = range(10, 20), visualisation = False, good = None, expert = None): 
    if visualisation:
        validation_env = gym.make('InvertedPendulum-v5', render_mode = "human")
    else:
        # validation_env = gym.make('InvertedPendulum-v5', reset_noise_scale = 0.01, max_episode_steps = 10000)
        validation_env = gym.make('Hopper-v5')
    
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