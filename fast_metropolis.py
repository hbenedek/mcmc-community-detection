import numpy as np
from parameters import * 
from utils import *
from tqdm import tqdm

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


def metropolis_step(x):
    #propose step
    i = np.random.randint(0, N)  
    #calculate acceptance prob
    acceptance = min(1, np.exp(- x[i] * x @ h[i]))   
    if random.random() < acceptance:
        #move on Metropolis chain
        x[i] = - x[i]


def run_metropolis(max_run, max_iter, fast_run=True):
    results = [] # list containing the results of MCMC results
    x_true = partition_to_vector('block') 
    for run in range(max_run):

        #initialize result object
        result = Result(run)
        
        #initial state
        x = np.random.choice((-1,1), N)    

        with tqdm(total=max_iter) as pbar:
            iter = 0
            while iter < max_iter:
                metropolis_step(x)
                current_overlap = calculate_overlap(x_true, x)
                if current_overlap == 1:
                    print(iter)
                result.overlaps1.append(current_overlap)
                iter = iter + 1
                pbar.update(1)
            
        print(f'run: {run}   overlap: {"{:.3f}".format(current_overlap)}')

        # save results 
        result.x = x
        results.append(result)

    return results

def houdayer_step(x1, x2):
    mask = (x1 != x2)
    potential_nodes = np.where(x1 != x2)

    #propose flip
    if mask.any():
        proposed_node = np.random.choice(potential_nodes[0], 1)
        H = G.subgraph(potential_nodes[0])
        target_nodes = list(nx.node_connected_component(H, int(proposed_node)))

        #perform houdayer move
       
        x1[target_nodes] = - x1[target_nodes]
        x2[target_nodes] = - x2[target_nodes]



def houdayer(max_run, max_iter, n, fast_run=True):
    results = [] # list containing the results of MCMC results
    x_true = partition_to_vector('block') 

    for run in range(max_run):
        
        #initialize result object
        result = Result(run)

        #initial state
        x1 = np.random.choice((-1,1), N)    
        x2 = np.random.choice((-1,1), N)  

        with tqdm(total=max_iter) as pbar:
            iter = 0
            while iter < max_iter:
            
                if iter % n == 0:
                      houdayer_step(x1, x2)
                #perform metropolis steps
                metropolis_step(x1)
                metropolis_step(x2)

                if not fast_run:
                    current_overlap1 = calculate_overlap(x_true,x1)
                    current_overlap2 = calculate_overlap(x_true,x2)
                    result.overlaps1.append(current_overlap1)
                    result.overlaps2.append(current_overlap2)

                iter = iter + 1
                pbar.update(1)

        print(f'run: {run}   overlap1: {"{:.3f}".format(current_overlap1)}  overlap2: {"{:.3f}".format(current_overlap2)}')

         # save results 
        result.x1 = x1
        result.x2 = x2
        results.append(result)


    return results