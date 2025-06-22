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
import sys 


def main():
    env = gym.make('Swimmer-v5', reset_noise_scale = 0.1)
    expert = TD3.load("Swimmer_models/td3_1m")
    # dt = DecisionTreeRegressor(random_state=42, max_leaf_nodes=10, max_depth=5)

    MAX_NODES = int(sys.argv[1])

    dt = DecisionTreeRegressor(random_state=42, max_leaf_nodes=MAX_NODES)

    best_dt = None 
    best_score = 0 

    # DAGGER parameters 
    dagger_iters = 30 # number of dagger iterations 
    rollouts = 5 # number of rollouts in each iteration
    beta = 0.5 # learning rate, at iteration i we choose expert policy beta ^ i times
    random_seed = 55 # seed for random python library 

    expansion_size = 200 # mutually exclusive with rollouts parameter, 
    # end iteration when dataset expanded by at least this number of samples

    random.seed(random_seed)
    D = [] 

    validation_scores = [] 

    print("Starting {} DAGGER iterations".format(dagger_iters))
    print("params:\nbeta = {}\trollouts = {}\tseed = {}\tmax_leaf = {}\n".format(beta, rollouts, random_seed, MAX_NODES)) 

    st_time = time.time() 

    for iter_num in range(dagger_iters):

        # beta_i = pow(beta, iter_num) 
        beta_i = 1 if iter_num == 0 else 0 
        cur_D_size = len(D) 

        print("dagger iteration = {}".format(iter_num)) 

        # rollout till the new dataset size expanded by at least expansion_size (5k) elements
        # or "rollouts" number of times 
        while len(D) < cur_D_size + expansion_size: 
        # for rollout in range(rollouts):

            obs, _states = env.reset(seed = random.randint(1, 10 ** 16)) 
            while True:

                choice = random.random()

                # # Query only expert for baseline 
                # beta_i = 1 
                # choice = 0 

                if choice < beta_i or iter_num == 0:
                    action, _states = expert.predict(obs)
                else:
                    action = dt.predict(obs.reshape(1, -1)).flatten()

                D.append((obs, expert.predict(obs)[0])) 
                obs, rewards, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    # obs, info = env.reset()
                    break 
        
        # Train DT on updated dataset D 

        X_states, y_actions = zip(*D)
        X_states = np.array(list(X_states)) 
        y_actions = np.array(list(y_actions)) 

        # print(X_states.shape) 
        # print(y_actions.shape) 

        print("fitting decision tree on {} samples".format(X_states.shape[0]))
        print("{}s elapsed".format(round(time.time() - st_time, 2)))

        dt.fit(X_states, y_actions)

        validation_scores.append(validate(dt))

        # can visualise the policy at each iteration 
        # validate(dt, seeds=[1], visualisation=True)

        # update the best policy 
        if(validation_scores[-1] > best_score):
            best_dt = copy.deepcopy(dt)
            best_score = validation_scores[-1] 

        print("validation score = {}\tbeta = {}\tnodes = {}".format(validation_scores[-1], beta_i, MAX_NODES))

        # if significant drop in performance, stop and save both trees 
        # if(best_score >8000 and validation_scores[-1] < 1000): 
        #     # significant drop in performance 
        #     with open('InvPen_Trees/debug_good.dump', 'wb') as file:
        #         pickle.dump(best_dt, file)
        #     with open('InvPen_Trees/debug_bad.dump', 'wb') as file:
        #         pickle.dump(dt, file)
        #     with open('InvPen_Trees/test_dataset.dump', 'wb') as file: 
        #         pickle.dump(D, file) 
        #     print("dumped both") 
        #     return 
    
    # Store the optimal policy 
    with open(f'results/Swimmer/{MAX_NODES}.dump', 'wb') as file:
        pickle.dump(best_dt, file) 

    with open(f'results/Swimmer/{MAX_NODES}_validations', 'w') as file:
        file.write(str(validation_scores))
    print(best_score) 

    # plot the learning curve 
    plt.figure()
    plt.plot(range(dagger_iters), validation_scores) 
    plt.savefig("dagger_plot")
    plt.show()

    # plt.figure(figsize=(12, 8)) 
    # plot_tree(dt, filled=True, feature_names= ['cart pos', 'angle', 'velocity', 'ang_vel'], class_names= ['force'])
    # plt.show()

    # plot the tree (tree_baseline.pdf file)
    # dot_data = export_graphviz(best_dt, out_file=None, feature_names=['cart pos', 'angle', 'velocity', 'ang_vel'], class_names=['force'], filled = True)
    # dot_data = export_graphviz(best_dt, out_file=None, feature_names=['ang0', 'ang1', 'ang2', 'velx', 'vely', 'a_vel0', 'a_vel1', 'a_vel2'], class_names=['torque1', 'torque2'], filled = True)
    dot_data = export_graphviz(best_dt, out_file=None, feature_names=['z_coord', 'ang0', 'ang1', 'ang2', 'ang3', 'vel_x', 'vel_z', 'ang_vel0', 'ang_vel1', 'ang_vel2', 'ang_vel3'], class_names=['torque1', 'torque2'], filled = True)

    render_tree(dot_data, MAX_NODES)

def render_tree(dot_data, MAX_NODES):
    graph = graphviz.Source(dot_data)
    graph.render(f"results/Swimmer/tree_{MAX_NODES}") 
    # graph.view()

"""
Validate the decision tree policy, on 10 fixed seeds 
Can also visualise the results 
returns the average of those 10 iterations 
"""
def validate(policy, seeds = range(10, 20), visualisation = False): 
    if visualisation:
        validation_env = gym.make('Swimmer-v5', render_mode = "human", reset_noise_scale = 0.01)
    else:
        # validation_env = gym.make('InvertedPendulum-v5', reset_noise_scale = 0.1, max_episode_steps = 10000)
        validation_env = gym.make('Swimmer-v5')
    
    result = 0 
    for seed in seeds:
        obs, _states = validation_env.reset(seed = seed)


        while True:
            action = policy.predict(obs.reshape(1, -1)).flatten()

            obs, rewards, terminated, truncated, info = validation_env.step(action)
            result += rewards 

            if terminated or truncated:
                # validation_env.close()
                break  
    
    result /= len(seeds) 
    validation_env.close() 
    return result 

if __name__ == "__main__":
    main() 