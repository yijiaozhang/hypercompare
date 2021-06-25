import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.community_embedding
from hypercomparison.link_prediction_community import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST
import numpy as np
import pandas as pd
import networkx as nx

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
beta = float(sys.argv[2])
temp_edges_path = sys.argv[4]
out_path = sys.argv[-1]

result_list = []

network = hypercomparison.networks.RealNetwork(network_name)
logger.info("Working on network {}, {}".format(network_name, beta))
splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
negative_edges = splitter.generate_negative_edges()
nx.write_edgelist(G, temp_edges_path)
network = hypercomparison.networks.ReloadedNetwork(temp_edges_path)
embeddings, superG = hypercomparison.community_embedding.Comm_Infomap().train(network) # community embedding based on Infomap algorithm
#embeddings, superG = hypercomparison.community_embedding.Comm_Louvain().train(network) # community embedding based on Louvain algorithm

test = LinkPredictionTask(
        test_edges, negative_edges, embeddings, network_name, superG, beta
    )
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, beta, roc_auc, aupr, average_precision, precision])

df = pd.DataFrame(result_list, columns=[
    'network_name', 'beta', 'roc_auc', 'aupr', 'average_precision', 'precision'
    ])

df.to_csv(out_path, index=None)
