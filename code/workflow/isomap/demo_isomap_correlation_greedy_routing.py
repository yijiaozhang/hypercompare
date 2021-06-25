import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.isomap
import hypercomparison.correlation_and_greedy_routing
import pandas as pd

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
dimensions = int(sys.argv[2])
out_path = sys.argv[-1]

result_list = []
network = hypercomparison.networks.RealNetwork(network_name)

logger.info("Working on network {} dimension {}".format(network_name, dimensions))
embeddings = hypercomparison.isomap.Isomap(dimension=dimensions).train(network)

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
