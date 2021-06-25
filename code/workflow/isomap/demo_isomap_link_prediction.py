import sys
import pandas as pd
import networkx as nx
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.isomap
from hypercomparison.link_prediction import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2])
temp_edges_path = sys.argv[3]

out_path = sys.argv[-1]
result_list = []

network = hypercomparison.networks.RealNetwork(network_name)
logger.info("Working on network {} dimension {}".format(network_name, dimensions))
splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
nx.write_edgelist(G, temp_edges_path)
network = hypercomparison.networks.ReloadedNetwork(temp_edges_path)
negative_edges = splitter.generate_negative_edges()
pos = hypercomparison.isomap.Isomap(dimension=dimensions).train(network)
test = LinkPredictionTask(
        test_edges, negative_edges, pos, name=network_name, proximity_function='euclidean_distance'
    )
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, dimensions, roc_auc, aupr, average_precision, precision])
df = pd.DataFrame(result_list, columns=['network_name', 'dimensions', 'roc_auc', 'aupr', 'average_precision', 'precision'])

df.to_csv(out_path, index=None)