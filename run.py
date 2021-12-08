from metropolis import *
from utils import *

results = metropolis(max_run, max_iter)

x_pred = estimate_posterior_mean(results)
x_true = partition_to_vector('block')

overlap = calculate_overlap(x_pred, x_true)

print(f'overlap: {overlap}')