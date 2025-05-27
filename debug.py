# import gymnasium as gym
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

    with open('InvPen_Trees/debug_good.dump', 'rb') as file:
        good = pickle.load(file) 
    with open('InvPen_Trees/debug_bad.dump', 'rb') as file:
        bad = pickle.load(file) 
    with open('InvPen_Trees/test_dataset.dump', 'rb') as file:
        dataset = pickle.load(file) 

    dataset = dataset[:-5000]
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

def render_tree(dot_data, name):
    graph = graphviz.Source(dot_data)
    graph.render(name) 
    # graph.view()

if __name__ == "__main__":
    main() 