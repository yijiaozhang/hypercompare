import sys
import pandas as pd
import hypercomparison.utils
import hypercomparison.networks
from hypercomparison.link_prediction import LinkPredictionTask
from hypercomparison.network_train_test_splitter import NetworkTrainTestSplitterWithMST

logger = hypercomparison.utils.get_logger(__name__)

mercator_coordinates_path = sys.argv[1]
test_edges_path = sys.argv[2]
negative_edges_path = sys.argv[3]
network_name = sys.argv[4]
out_path = sys.argv[-1]

result_list = []

logger.info("Working on network {}".format(network_name))
mercator_coordinates = pd.read_csv(
    mercator_coordinates_path, comment='#', 
    delim_whitespace=True, names=['node_id', 'kappa', 'angular', 'radial']
    )
mercator_coordinates = mercator_coordinates[['node_id', 'angular', 'radial']]
embeddings = mercator_coordinates.set_index('node_id').T.to_dict('list')
embeddings = {str(k):v for k,v in embeddings.items()}

test_edges = pd.read_csv(test_edges_path, dtype='object')
negative_edges = pd.read_csv(negative_edges_path, dtype='object')
test_edges = test_edges.values.tolist()
negative_edges = negative_edges.values.tolist()

test = LinkPredictionTask(
        test_edges, negative_edges, embeddings, name=network_name, proximity_function='hyperbolic_distance'
    )
roc_auc, aupr, average_precision, precision = test.do_link_prediction()
result_list.append([network_name, roc_auc, aupr, average_precision, precision])
df = pd.DataFrame(result_list, columns=['network_name', 'roc_auc', 'aupr', 'average_precision', 'precision'])

df.to_csv(out_path, index=None)