import sys
import numpy as np
import pandas as pd
import networkx as nx
from main import *
import torch
import hypercomparison.utils
import hypercomparison.networks
from hypercomparison.link_prediction import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
poincare_coordinates_path = sys.argv[2]
temp_edges_path = sys.argv[3]
test_edges_path = sys.argv[4]
negative_edges_path = sys.argv[5]
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

network = hypercomparison.networks.RealNetwork(network_name)

logger.info("Working on network {}".format(network_name))

splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
negative_edges = splitter.generate_negative_edges()
nx.write_edgelist(G, temp_edges_path)
test_edges_df = pd.DataFrame(test_edges)
negative_edges_df = pd.DataFrame(negative_edges)
test_edges_df.to_csv(test_edges_path, index=None)
negative_edges_df.to_csv(negative_edges_path, index=None)
features = torch.DoubleTensor(nx.to_numpy_matrix(G))
poincare_coord, _ = compute_poincare_maps(
    features, None, poincare_coordinates_path, 
    mode = 'KNN', epochs=1000, earlystop=0.00001, cuda=0)

poincare_coordinates = pd.read_csv(poincare_coordinates_path + '.csv', header=None)
poincare_coordinates = np.array(poincare_coordinates)
embeddings = convert_embeddings(network, poincare_coordinates)
test = LinkPredictionTask(
        test_edges, negative_edges, embeddings, name=network_name, proximity_function='poincare_distance'
    )
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, roc_auc, aupr, average_precision, precision])
df = pd.DataFrame(result_list, columns=['network_name', 'roc_auc', 'aupr', 'average_precision', 'precision'])
df.to_csv(out_path, index=None)