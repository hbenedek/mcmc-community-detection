import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random  
from parameters import * 
from tqdm import tqdm

#####################    METROPOLIS    #####################  

class Result():
    def __init__(self, name):
        self.name = name
        self.overlaps = []
        self.x = None
 
def partition_to_vector(label):
    x = np.ones(N)
    for i in range(N):
        x[i] = G.nodes[i][label]
    x[x == 0] = -1
    return x

def calculate_h(node_i, node_j):
    if G.has_edge(node_i, node_j):
        return np.log(a) - np.log(b)
    else:
        return np.log(1 - a / N) - np.log(1 - b / N)

def calculate_local_hamiltonian(node):
    return sum([calculate_h(node, neighbor) * G.nodes[neighbor]['estimate'] for neighbor in G.neighbors(node)])

def calculate_stationary_ratio(node):
    hamiltonian = calculate_local_hamiltonian(node)
    return np.exp(- G.nodes[node]['estimate'] * hamiltonian)

def metropolis_step():
    #propose step
    proposed_node = np.random.randint(0, N)
          
    #calculate acceptance prob
    stationary_ratio = calculate_stationary_ratio(proposed_node)
    acceptance = min(1, stationary_ratio)
           
    if random.random() < acceptance:
        #move on Metropolis chain
        G.nodes[proposed_node]['estimate'] = - G.nodes[proposed_node]['estimate']


def run_monte_carlo(max_run, max_iter, fast_run=True):
    results = [] # list containing the results of MCMC results
    x_true = partition_to_vector('block') 

    for run in range(max_run):
        #initial state
        x = np.random.choice((-1,1), N)    
        for node in range(N): 
            G.nodes[node]['estimate'] = x[node]

        #initialize result object
        result = Result(run)
        # initial_overlap = calculate_overlap(x, x_true)
        # result.overlaps.append(initial_overlap)

        with tqdm(total=max_iter) as pbar:
            iter = 0
            while iter < max_iter:

                metropolis_step()

                if not fast_run:
                    x_pred = partition_to_vector('estimate')
                    current_overlap = calculate_overlap(x_true,x_pred)
                    result.overlaps.append(current_overlap)

                iter = iter + 1
                pbar.update(1)
                
        x_pred = partition_to_vector('estimate')
        overlap = calculate_overlap(x_true, x_pred)
        print(f'run: {run}   overlap: {"{:.3f}".format(overlap)}')

        # save results 
        result.x = partition_to_vector('estimate')
        results.append(result)

    return results
     

def estimate_posterior_mean(results):
    estimator = np.zeros(N)
    for result in results:
        estimator = estimator + result.x
    return np.sign(estimator) + (estimator == 0) 
