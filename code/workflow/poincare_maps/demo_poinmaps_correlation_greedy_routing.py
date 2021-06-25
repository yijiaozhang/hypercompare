import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.correlation_and_greedy_routing
from main import *
import torch
import numpy as np
import pandas as pd
import networkx as nx

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
poincare_coordinates_path = sys.argv[2]
out_path = sys.argv[-1]

result_list = []

def convert_embeddings(network, poincare_coord):
    network.index_nodes()
    network_size = len(network.G.nodes())
    result = []
    for i in range(network_size):
        node = network.id2node[i]
        result.append([
            node, poincare_coord[i][0], poincare_coord[i][1]])
    result_df = pd.DataFrame(result, columns=['node', 'x', 'y'])
    embeddings = result_df.set_index('node').T.to_dict('list')
    embeddings = {str(k):v for k,v in embeddings.items()}
    return embeddings

logger.info("Working on network {}, poincare maps".format(network_name))

network = hypercomparison.networks.RealNetwork(network_name)

features = torch.DoubleTensor(nx.to_numpy_matrix(network.G))
poincare_coord, _ = compute_poincare_maps(
    features, None, poincare_coordinates_path, 
    mode = 'KNN', epochs=1000, earlystop=0.00001, cuda=0)
poincare_coordinates = pd.read_csv(poincare_coordinates_path + '.csv', header=None)
poincare_coordinates = np.array(poincare_coordinates)
embeddings = convert_embeddings(network, poincare_coordinates)

logger.info("Working on network {}".format(network_name))
all_tasks = hypercomparison.correlation_and_greedy_routing.AllTasks(network, embeddings, distance_func='poincare')

rate, length, efficiency, stretch, score = all_tasks.greedy_routing()
pearson_correlation, _, spearman_correlation, _ = all_tasks.calculate_correlation()
absolute_error, relative_error = all_tasks.calculate_error()
result_list.append([network_name, rate, length, efficiency, stretch, score,
        pearson_correlation, spearman_correlation, absolute_error, relative_error])

df = pd.DataFrame(result_list, columns=[
    'network_name', 'gr_rate', 'gr_length', 'gr_efficiency','gr_stretch', 'gr_score',
    'pearson_correlation', 'spearman_correlation', 'absolute_error', 'relative_error'])

df.to_csv(out_path, index=None)