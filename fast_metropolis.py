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


def run_metropolis_fast(max_run, max_iter, fast_run=True):
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

                if not fast_run:
                    current_overlap = calculate_overlap(x_true, x)
                    result.overlaps1.append(current_overlap)

                iter = iter + 1
                pbar.update(1)
            
        overlap = calculate_overlap(x_true, x)
        print(f'run: {run}   overlap: {"{:.3f}".format(overlap)}')

        # save results 
        results.append(result)

    return results