from utils import *

a = 1
b = 1
N = 1000

prior = prior(N)
G = generate_sbm_graph(prior, N - prior, a, b)

max_run = 100
max_iter = 10000