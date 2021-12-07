
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random  
from parameters import * 

#####################    METROPOLIS    #####################  


def generate_proposed_step(N):
    node = random.randint(0, N)
    color = 1 if random.random() > 1/2 else -1
    return node, color 

def calculate_h(node_i, node_j):
    e = 1 if G.has_edge(node_i, node_j) else 0
    return 1/2 * (e * (np.log(a) - np.log(b)) + (1 - e) * np.log(1 - a/N) - np.log(1 - a/N))

def calculate_stationary_ratio(x, proposed_step):
    # for Neighbour(node) calculate prod e^{h_ij x_i x_j}
    pass

def metropolis(max_run, max_iter):
    metropolis_simulations = []
    for run in range(max_run):
        #initial state
        for iter in range(max_iter):
            pass
            #generate step

            #calculate acceptence prob

            #decide wheter we accept

            #move on Metropolis chain
        
    return metropolis_simulations 

def estimate_posterior_mean(runs):
    return np.sign(sum(runs))