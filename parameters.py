from utils import *

a = 10
b = 1
N = 50

prior = prior(N)
G = generate_sbm_graph(prior, N - prior, a, b)

max_run = 10
max_iter = 1000