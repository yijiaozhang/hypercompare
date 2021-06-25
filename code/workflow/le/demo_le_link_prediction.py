import sys
import pandas as pd
import networkx as nx
from sklearn.manifold import SpectralEmbedding
import hypercomparison.utils
import hypercomparison.networks
from hypercomparison.link_prediction import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2])

out_path = sys.argv[-1]
result_list = []

network = hypercomparison.networks.RealNetwork(network_name)
network.index_nodes()
if dimensions > len(network.G.nodes()):
    dimensions = len(network.G.nodes())
logger.info("Working on network {} dimension {}".format(network_name, dimensions))
splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
negative_edges = splitter.generate_negative_edges()
adjacency_matrix = nx.adjacency_matrix(G).todense()
le_coordinates = SpectralEmbedding(n_components=dimensions, affinity='precomputed').fit_transform(adjacency_matrix)
embeddings = {str(network.id2node[i]): le_coordinates[i] for i in range(len(network.id2node))}
test = LinkPredictionTask(
        test_edges, negative_edges, embeddings, name=network_name, proximity_function='euclidean_distance'
    )
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, dimensions, roc_auc, aupr, average_precision, precision])
df = pd.DataFrame(result_list, columns=['network_name', 'dimensions', 'roc_auc', 'aupr', 'average_precision', 'precision'])

df.to_csv(out_path, index=None)