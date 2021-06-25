import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.node2vec_HOPE
import hypercomparison.correlation_and_greedy_routing

import networkx as nx
import numpy as np
import pandas as pd

# Force numpy to only use single thread in linear algebra
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2]) 

out_path = sys.argv[-1]

result_list = []
network = hypercomparison.networks.RealNetwork(network_name)
adjacency_matrix = nx.to_numpy_array(network.G)
e = np.linalg.eigvals(adjacency_matrix)
beta=1/max(e).real - 0.001
logger.info("network {} dimension {}, beta calculated".format(network_name, dimensions))
if dimensions > len(network.G.nodes()):
    dimensions = len(network.G.nodes())

logger.info("Working on network {} dimension {}".format(network_name, dimensions))
embeddings = hypercomparison.node2vec_HOPE.HOPE(dimension=dimensions, beta=beta).train(network.G)
all_tasks = hypercomparison.correlation_and_greedy_routing.AllTasks(network, embeddings)
rate, length, efficiency, stretch, score = all_tasks.greedy_routing()
pearson_correlation, _, spearman_correlation, _ = all_tasks.calculate_correlation()
absolute_error, relative_error = all_tasks.calculate_error()
result_list.append([
    network_name, dimensions, beta, 
    rate, length, efficiency, stretch, score, 
    pearson_correlation, spearman_correlation, absolute_error, relative_error])

df = pd.DataFrame(result_list, columns=[
    'network_name', 'dimensions', 'beta', 
    'gr_rate', 'gr_length', 'gr_efficiency', 'gr_stretch', 'gr_score', 
    'pearson_correlation', 'spearman_correlation', 'absolute_error', 'relative_error'
    ])

df.to_csv(out_path, index=None)
