import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fast_metropolis import *
from parameters import *
from utils import *


for i in [500, 200, 100, 50]:
    print(f"###   HOUDAYER with flip {i}   ###")
    mixed_results = houdayer(max_run, max_iter, i, fast_run=False)
    m, l, u = get_averages_and_CI(mixed_results, max_iter)
    dump_pickle([m, l, u], f"./results/houdayer_{i}.pickle")



#mixed_results_200 = houdayer(max_run, max_iter, 200, fast_run=False)
#dump_pickle(mixed_results_200, "./results/mixed_results_200.pickle")

#mixed_results_100 = houdayer(max_run, max_iter, 100, fast_run=False)
#dump_pickle(mixed_results_100, "./results/mixed_results_100.pickle")

#mixed_results_50 = houdayer(max_run, max_iter, 50, fast_run=False)
#dump_pickle(mixed_results_50, "./results/mixed_results_50.pickle")



