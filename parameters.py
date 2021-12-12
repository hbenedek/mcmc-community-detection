from utils import *

a = 10
b = 1
N = 50

prior = prior(N)
print(f'community sizes: {prior} - {N-prior}')

G = generate_sbm_graph(prior, N - prior, a, b)

max_run = 100
max_iter = 500