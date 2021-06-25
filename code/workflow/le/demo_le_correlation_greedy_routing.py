import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.correlation_and_greedy_routing
import pandas as pd
import networkx as nx
from sklearn.manifold import SpectralEmbedding

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2])
out_path = sys.argv[-1]

result_list = []
network = hypercomparison.networks.RealNetwork(network_name)
network.index_nodes()
adjacency_matrix = nx.adjacency_matrix(network.G).todense()
if dimensions > len(network.G.nodes()):
    dimensions = len(network.G.nodes())
logger.info("Working on network {} dimension {}".format(network_name, dimensions))
le_coordinates = SpectralEmbedding(n_components=dimensions, affinity='precomputed').fit_transform(adjacency_matrix)
embeddings = {str(network.id2node[i]): le_coordinates[i] for i in range(len(network.id2node))}

all_tasks = hypercomparison.correlation_and_greedy_routing.AllTasks(network, embeddings)
rate, length, efficiency, stretch, score = all_tasks.greedy_routing()
pearson_correlation, _, spearman_correlation, _ = all_tasks.calculate_correlation()
absolute_error, relative_error = all_tasks.calculate_error()
result_list.append([
    network_name, dimensions, rate, length, efficiency, stretch, score, 
    pearson_correlation, spearman_correlation, absolute_error, relative_error])

df = pd.DataFrame(result_list, columns=[
    'network_name', 'dimensions', 'gr_rate', 'gr_length', 'gr_efficiency', 'gr_stretch', 'gr_score', 
    'pearson_correlation', 'spearman_correlation', 'absolute_error', 'relative_error'
    ])

df.to_csv(out_path, index=None)
