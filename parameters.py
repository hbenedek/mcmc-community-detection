from utils import *

a = 40
b = 15
N = 1000

prior = prior(N)
G = generate_sbm_graph(prior, N - prior, a, b)