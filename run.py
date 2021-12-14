from metropolis import *
from utils import *

check_ab_condition(a, b)

results = run_metropolis(max_run, max_iter, fast_run=False)

x_pred = estimate_posterior_mean(results)
x_true = partition_to_vector('block')

overlap = calculate_overlap(x_pred, x_true)

print(f'Bayes estimator overlap: {"{:.3f}".format(overlap)}')
