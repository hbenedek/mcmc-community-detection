import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random  
from parameters import * 
from tqdm import tqdm

#####################    METROPOLIS    #####################  

def partition_to_vector(label):
    x = np.ones(N)
    for i in range(N):
        x[i] = G.nodes[i][label]
    x[x == 0] = -1
    return x

def generate_proposed_step(G):
    node = random.randint(0, N)
    return node, G.nodes[node]['estimate'] * -1

def calculate_h(node_i, node_j):
    e = 1 if G.has_edge(node_i, node_j) else 0
    return 1/2 * (e * (np.log(a) - np.log(b)) + (1 - e) * np.log(1 - a/N) - np.log(1 - a/N))

def calculate_local_hamiltonian(node):
    hamiltonian = 0
    for n in G.neighbors(node):
        hamiltonian += calculate_h(node, n) * G.nodes[n]['estimate']
    return hamiltonian



def calculate_stationary_ratio(node):
    hamiltonian = calculate_local_hamiltonian(node)
    return np.exp(- 2 * G.nodes[node]['estimate'] * hamiltonian)
    # for Neighbour(node) calculate prod e^{h_ij x_i x_j}
    

def metropolis(max_run, max_iter):
    metropolis_simulations = [] # list containing the results of MCMC runs
    elements = list(G.nodes) #list of nodes for silmulating the sampling
    elements.append('REJECT') 

    for run in range(max_run):
        print(f' *** run {run} ***')
        #initial state
        x = np.random.choice((-1,1), N)
        
        for node in range(N): 
            G.nodes[node]['estimate'] = x[node]

        for iter in tqdm(range(max_iter)):
            #generate step
            probabilities = []
            for proposed_node in G.nodes:
                #calculate acceptence prob
                stationary_ratio = calculate_stationary_ratio(proposed_node)
                acceptence = min(1, stationary_ratio)
                transition_probability = 1/N * acceptence
                probabilities.append(transition_probability)
            
            #probability of rejecting the move
            probabilities.append(1 - sum(probabilities))
           
            #pick a neighbour according to prob distribution
            step = np.random.choice(elements , 1, p=probabilities)
            

            #move on Metropolis chain (set new coloring)
            if step != 'REJECT':
                G.nodes[int(step)]['estimate'] = -1 * G.nodes[int(step)]['estimate']

        result = partition_to_vector('estimate')
        metropolis_simulations.append(result)

    return metropolis_simulations
     

def estimate_posterior_mean(runs):
    return np.sign(sum(runs))

