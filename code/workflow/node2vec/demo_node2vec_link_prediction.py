import sys
import pandas as pd
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.embedding_centered
from hypercomparison.link_prediction import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2])
walk_length = int(sys.argv[3])
out_path = sys.argv[-1]

result_list = []
network = hypercomparison.networks.RealNetwork(network_name) 
if dimensions > len(network.G.nodes()): 
    dimensions = len(network.G.nodes())  
logger.info("Working on network {} dimension {}, walk length {}".format(network_name, dimensions, walk_length))
splitter = NetworkTrainTestSplitterWithMST(network.G)
G, test_edges = splitter.train_test_split()
negative_edges = splitter.generate_negative_edges()
pos = hypercomparison.embedding_centered.Node2vec(dimension=dimensions, walk_length=walk_length).train(G)
test = LinkPredictionTask(test_edges, negative_edges, pos, name=network_name)
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, dimensions, walk_length, roc_auc, aupr, average_precision, precision])
df = pd.DataFrame(result_list, columns=[
    'network_name', 'dimensions', 'walk_length', 'roc_auc', 'aupr', 'average_precision', 'precision'])

df.to_csv(out_path, index=None)