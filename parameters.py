from utils import *

a = 5.9
b = 0.1
N = 1000

prior = prior(N)
print(f'community sizes: {prior} - {N-prior}')

G = generate_sbm_graph(prior, N - prior, a, b)
h = init_h(G, N, a, b)

max_run = 10
max_iter = 10000