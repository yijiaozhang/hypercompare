import sys
import numpy as np
import pandas as pd
import networkx as nx

import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.embedding_centered
from hypercomparison.link_prediction import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST

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
if dimensions > len(network.G.nodes()):
    dimensions = len(network.G.nodes())
    
logger.info("Working on network {} dimension {}".format(network_name, dimensions))
#split test and negative edges
splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
negative_edges = splitter.generate_negative_edges()
#calculate embeddings
adjacency_matrix = nx.to_numpy_array(G)
e = np.linalg.eigvals(adjacency_matrix)
beta=1/max(e).real - 0.001
logger.info("network {} dimension {}, beta calculated".format(network_name, dimensions))
embeddings = hypercomparison.embedding_centered.HOPE(dimension=dimensions, beta=beta).train(G)
#perform link prediction
test = LinkPredictionTask(test_edges, negative_edges, embeddings, name=network_name) 
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, dimensions, beta, roc_auc, aupr, average_precision, precision])
df = pd.DataFrame(result_list, columns=['network_name', 'dimensions', 'beta', 'roc_auc', 'aupr', 'average_precision', 'precision'])

df.to_csv(out_path, index=None)