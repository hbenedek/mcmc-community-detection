from utils import *

a = 40
b = 15
N = 100

prior = prior(N)
G = generate_sbm_graph(prior, N - prior, a, b)

max_run = 10
max_iter = 500