from utils import *
import numpy as np

a = 5.9
b = 0.1
N = 1000

prior = prior(N)
print(f'community sizes: {prior} - {N-prior}')

G = generate_sbm_graph(prior, N - prior, a, b)
A = networkx_to_adjacency(G).toarray()
h = adjacency_to_h(A, N, a, b)

max_run = 100
max_iter = 100000

flip = 200