import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random  
from parameters import * 
from tqdm import tqdm

#####################    METROPOLIS    #####################  

hamiltonian = []

class Result():
    def __init__(self, name):
        self.name = name
        self.overlaps1 = []
        self.overlaps2 = []
        self.x = None
 
def partition_to_vector(label):
    x = np.ones(N)
    for i in range(N):
        x[i] = G.nodes[i][label]
    x[x == 0] = -1
    return x

def calculate_local_hamiltonian(node, estimate):
    nodes = set(G.nodes) - set([node])
    return sum([h[node][n] * G.nodes[n][estimate] for n in list(nodes)])

def calculate_stationary_ratio(node, estimate):
    hamiltonian = calculate_local_hamiltonian(node, estimate)
    return np.exp(- G.nodes[node][estimate] * hamiltonian)

def metropolis_step(estimate):
    #propose step
    proposed_node = np.random.randint(0, N)
          
    #calculate acceptance prob
    stationary_ratio = calculate_stationary_ratio(proposed_node, estimate)
    acceptance = min(1, stationary_ratio)
           
    if random.random() < acceptance:
        #move on Metropolis chain
        G.nodes[proposed_node][estimate] = - G.nodes[proposed_node][estimate]


def run_metropolis(max_run, max_iter, fast_run=True):
    results = [] # list containing the results of MCMC results
    x_true = partition_to_vector('block') 

    for run in range(max_run):

        #initialize result object
        result = Result(run)

        #initial state
        x = np.random.choice((-1,1), N)    
        for node in range(N): 
            G.nodes[node]['estimate'] = x[node]
    

        with tqdm(total=max_iter) as pbar:
            iter = 0
            while iter < max_iter:

                metropolis_step('estimate')

                if not fast_run:
                    x_pred = partition_to_vector('estimate')
                    current_overlap = calculate_overlap(x_true,x_pred)
                    result.overlaps1.append(current_overlap)

                iter = iter + 1
                pbar.update(1)
                
        x_pred = partition_to_vector('estimate')
        overlap = calculate_overlap(x_true, x_pred)
        print(f'run: {run}   overlap: {"{:.3f}".format(overlap)}')

        # save results 
        result.x = partition_to_vector('estimate')
        results.append(result)

    return results

def houdayer_step():
    potential_nodes = []
    for node in range(N): 
        local_overlap = G.nodes[node]['estimate1'] * G.nodes[node]['estimate2']
        G.nodes[node]['local_overlap'] = local_overlap
        if local_overlap == -1:
            potential_nodes.append(node)

    #propose flip
    if len(potential_nodes) > 0:
        proposed_node = np.random.choice(potential_nodes, 1)
        H = G.subgraph(potential_nodes)
        target_nodes = nx.node_connected_component(H, int(proposed_node))

        #perform houdayer move
        for node in target_nodes:
            G.nodes[node]['estimate1'] = - G.nodes[node]['estimate1']
            G.nodes[node]['estimate2'] = - G.nodes[node]['estimate2']

def houdayer(max_run, max_iter, n, fast_run=True):
    results = [] # list containing the results of MCMC results
    x_true = partition_to_vector('block') 

    for run in range(max_run):
        
        #initialize result object
        result = Result(run)

        #initial state
        x1 = np.random.choice((-1,1), N)    
        x2 = np.random.choice((-1,1), N)  

        for node in range(N): 
            G.nodes[node]['estimate1'] = x1[node]
            G.nodes[node]['estimate2'] = x2[node]

        with tqdm(total=max_iter) as pbar:
            iter = 0
            while iter < max_iter:
                #compute local overlap 
              
                if iter % n == 0:
                      houdayer_step()
                #perform metropolis steps
                metropolis_step('estimate1')
                metropolis_step('estimate2')

                if not fast_run:
                    x1_pred = partition_to_vector('estimate1')
                    x2_pred = partition_to_vector('estimate2')
                    current_overlap1 = calculate_overlap(x_true,x1_pred)
                    current_overlap2 = calculate_overlap(x_true,x2_pred)
                    result.overlaps1.append(current_overlap1)
                    result.overlaps2.append(current_overlap2)


                iter = iter + 1
                pbar.update(1)

        # save results 
        result.x1 = partition_to_vector('estimate1')
        result.x2 = partition_to_vector('estimate2')
        results.append(result)

        overlap1 = calculate_overlap(x_true, result.x1)
        overlap2 = calculate_overlap(x_true, result.x2)
        print(f'run: {run}   overlap1: {"{:.3f}".format(overlap1)}  overlap2: {"{:.3f}".format(overlap2)}')


    return results
    

def estimate_posterior_mean(results, algorithm):
    estimator1 = np.zeros(N)
    estimator2 = np.zeros(N)
    if algorithm == 'metropolis':
        for result in results:
            estimator1 = estimator1 + result.x
        return np.sign(estimator1) + (estimator1 == 0) 
    if algorithm == 'houdayer':
        for result in results:
            estimator1 = estimator1 + result.x1
            estimator2 = estimator2 + result.x2
        return np.sign(estimator1) + (estimator1 == 0), np.sign(estimator2) + (estimator2 == 0)

