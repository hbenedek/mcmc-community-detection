import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fast_metropolis import *
from parameters import *
from utils import *

results = run_metropolis(max_run, max_iter)
results = houdayer(max_run, max_iter, flip, fast_run=False)
#x1, x2 = estimate_posterior_mean(results, 'houdayer')
#x_true = partition_to_vector('block')

#overlap1 = calculate_overlap(x1, x_true)
#overlap2 = calculate_overlap(x2, x_true)

#print(f'Bayes overlap1: {"{:.3f}".format(overlap1)}')
#print(f'Bayes overlap2: {"{:.3f}".format(overlap2)}')