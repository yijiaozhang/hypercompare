import sys
import numpy as np
import pandas as pd
import networkx as nx

import hypercomparison.utils
import hypercomparison.networks
from hypercomparison.link_prediction import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST

from rpy2.robjects.packages import importr
hydra = importr('hydra')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2])
temp_edges_path = sys.argv[3]

out_path = sys.argv[-1]
result_list = []

network = hypercomparison.networks.RealNetwork(network_name)
network.index_nodes()
if dimensions > len(network.G.nodes):
    dimensions = len(network.G.nodes)

logger.info("Working on network {} dimension {}".format(network_name, dimensions))
splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
negative_edges = splitter.generate_negative_edges()
nx.write_edgelist(G, temp_edges_path)

network = hypercomparison.networks.ReloadedNetwork(temp_edges_path)
network.index_nodes()
network.generate_shortest_path_length_matrix()

pos = hydra.hydra(network.shortest_path_length_matrix, dimensions)
directional = np.array(pos.rx2('directional'))
r = np.array(pos.rx2('r'))
temp_pos = [list(directional[i]*r[i]) for i in range(len(network.G.nodes))]
temp_pos = np.array(temp_pos)
embeddings = {str(network.id2node[i]): temp_pos[i] for i in range(len(network.id2node))}

test = LinkPredictionTask(test_edges, negative_edges, embeddings, name=network_name, proximity_function='poincare_distance')
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, dimensions, roc_auc, aupr, average_precision, precision])
df = pd.DataFrame(result_list, columns=['network_name', 'dimensions', 'roc_auc', 'aupr', 'average_precision', 'precision'])

df.to_csv(out_path, index=None)