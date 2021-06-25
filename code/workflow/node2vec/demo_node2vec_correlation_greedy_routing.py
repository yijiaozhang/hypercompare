import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.node2vec_HOPE
import hypercomparison.correlation_and_greedy_routing
import pandas as pd

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
embeddings = hypercomparison.node2vec_HOPE.Node2vec(dimension=dimensions, walk_length=walk_length).train(network.G)

all_tasks = hypercomparison.correlation_and_greedy_routing.AllTasks(network, embeddings)
rate, length, efficiency, stretch, score = all_tasks.greedy_routing()
pearson_correlation, _, spearman_correlation, _ = all_tasks.calculate_correlation()
absolute_error, relative_error = all_tasks.calculate_error()
result_list.append([
    network_name, dimensions, walk_length, 
    rate, length, efficiency, stretch, score, 
    pearson_correlation, spearman_correlation, absolute_error, relative_error])

df = pd.DataFrame(result_list, columns=[
    'network_name', 'dimensions', 'walk_length',
    'gr_rate', 'gr_length', 'gr_efficiency', 'gr_stretch', 'gr_score', 
    'pearson_correlation', 'spearman_correlation', 'absolute_error', 'relative_error'
    ])

df.to_csv(out_path, index=None)
