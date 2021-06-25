import sys
import hypercomparison.utils
import hypercomparison.networks
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST
import networkx as nx
import pandas as pd
import mercator

logger = hypercomparison.utils.get_logger(__name__)
network_name = sys.argv[1]
mercator_coordinates_path = sys.argv[2]
test_edges_path = sys.argv[3]
negative_edges_path = sys.argv[4]
temp_edge_path = sys.argv[5]

network = hypercomparison.networks.RealNetwork(network_name)
splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
nx.write_edgelist(G, temp_edge_path)
negative_edges = splitter.generate_negative_edges()
test_edges_df = pd.DataFrame(test_edges)
negative_edges_df = pd.DataFrame(negative_edges)
test_edges_df.to_csv(test_edges_path, index=None)
negative_edges_df.to_csv(negative_edges_path, index=None)

mercator.embed(temp_edge_path, output_name=mercator_coordinates_path)