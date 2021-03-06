import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats
import random
import seaborn as sns
import pickle

def prior(N):
    return np.random.binomial(N, 1/2)

def generate_sbm_graph(size_1, size_2, a, b):
    sizes = [size_1, size_2]
    N = sum(sizes)
    probes = [[a / N, b / N], [b / N, a / N]]
    return nx.generators.community.stochastic_block_model(sizes, probes)

def init_h(G, N, a, b):
    h = np.zeros((N, N))
    edge1 = np.log(a) - np.log(b)
    edge0 = np.log(1 - a / N) - np.log(1 - b / N)
    print(f'edge energy: {"{:.3f}".format(edge1)}, non-edge energy: {"{:.3f}".format(edge0)}')
    for i in range(N):
        for j in range(i):
            if G.has_edge(i, j):
                h[i][j] = edge1
                h[j][i] = edge1
            else:
                h[i][j] = edge0
                h[j][i] = edge0
    return h

def adjacency_to_h(A, N, a, b):
    h = np.zeros((N, N))
    h[A==1] = np.log(a) - np.log(b)
    h[A==0] = np.log(1 - a / N) - np.log(1 - b / N)
    return h


def draw_graph(G, x):
    """
    Plots graph with colored nodes according to the coloring x

    Parameters
    ----------
    G : nx.Graph 
    x : np.ndarray - coloring vector of size {-1,1}^N 
    """
    colors = []
    for node in G.nodes():
        if x[node] == 1:
            colors.append('blue')
        else:
            colors.append('red')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40,40))
    #pos = nx.nx_pydot.graphviz_layout(G)
    pos = nx.spring_layout(G)
    nx.draw(G, node_color=colors, pos=pos, node_size=70, ax=ax)

    
def calculate_overlap(x_true, x_pred):
    return np.abs(1 / len(x_true) * x_true.T @ x_pred)


def check_ab_condition(a, b):
    if (a - b)**2 <= 2 * (a+b):
        raise ValueError('The choice of (a, b) are not valid, the variables must satisfy (a - b)^2 <= 2*(a+b)')


def plot_overlap_evolution(overlaps):
    N = len(overlaps)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    sns.lineplot(np.arange(N), np.array(overlaps))


def dump_pickle(data, file):
    print("dumping results")
    with open(file, 'wb') as handle:
        pickle.dump(data , handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_pickle(file):
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    return data


def adjacency_to_networkx(A):
    return nx.convert_matrix.from_numpy_matrix(A)

def networkx_to_adjacency(G):
    return nx.linalg.graphmatrix.adjacency_matrix(G)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def get_averages_and_CI(results, max_iter):
    lower = []
    mean = []
    upper = []
    for i in range(max_iter):
        data1 = []
        for result in results:
            data1.append(result.overlaps1[i])
        m, l, u = mean_confidence_interval(data1, confidence=0.95)
        lower.append(l)
        mean.append(m)
        upper.append(u)
    return mean, lower, upper