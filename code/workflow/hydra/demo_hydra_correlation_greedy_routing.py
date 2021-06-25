import sys
import numpy as np
import pandas as pd
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.correlation_and_greedy_routing

from rpy2.robjects.packages import importr
hydra = importr('hydra')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2])

out_path = sys.argv[-1]

result_list = []
network = hypercomparison.networks.RealNetwork(network_name)
network.index_nodes()
network.generate_shortest_path_length_matrix()

if dimensions > len(network.G.nodes):
    dimensions = len(network.G.nodes)

pos = hydra.hydra(network.shortest_path_length_matrix, dimensions)
directional = np.array(pos.rx2('directional'))
r = np.array(pos.rx2('r'))
temp_pos = [list(directional[i]*r[i]) for i in range(len(network.G.nodes))]
temp_pos = np.array(temp_pos)
embeddings = {str(network.id2node[i]): temp_pos[i] for i in range(len(network.id2node))}

logger.info("Working on network {} dimension {}".format(network_name, dimensions))
all_tasks = hypercomparison.correlation_and_greedy_routing.AllTasks(network, embeddings, distance_func='poincare')
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