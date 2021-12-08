import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def prior(N):
    return np.random.binomial(N, 1/2)

def generate_sbm_graph(size_1, size_2, a, b):
    sizes = [size_1, size_2]
    N = sum(sizes)
    probs = [[a / N, b / N], [b / N, a / N]]
    return nx.generators.community.stochastic_block_model(sizes, probs)

def draw_graph(G):
    colors = []
    for node in G.nodes():
        if node in G.graph['partition'][0]:
            colors.append('blue')
        else:
            colors.append('red')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40,40))
    #pos = nx.nx_pydot.graphviz_layout(G)
    pos = nx.spring_layout(G)
    nx.draw(G, node_color=colors, pos=pos, node_size=70, ax=ax)

    
def calculate_overlap(x_true, x_pred):
    return np.abs(1 / len(x_true) * x_true @ x_pred)