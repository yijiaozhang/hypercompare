import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.correlation_and_greedy_routing
import pandas as pd

logger = hypercomparison.utils.get_logger(__name__)
mercator_coordinates_path = sys.argv[1]
network_name = sys.argv[2]
out_path = sys.argv[-1]

result_list = []
network = hypercomparison.networks.RealNetwork(network_name)

mercator_coordinates = pd.read_csv(
    mercator_coordinates_path, comment='#', 
    delim_whitespace=True, names=['node_id', 'kappa', 'angular', 'radial']
    )
mercator_coordinates = mercator_coordinates[['node_id', 'angular', 'radial']]
embeddings = mercator_coordinates.set_index('node_id').T.to_dict('list')
embeddings = {str(k):v for k,v in embeddings.items()}

logger.info("Working on network {}".format(network_name))
all_tasks = hypercomparison.correlation_and_greedy_routing.AllTasks(network, embeddings, distance_func='hyperbolic')
rate, length, efficiency, stretch, score = all_tasks.greedy_routing()
pearson_correlation, _, spearman_correlation, _ = all_tasks.calculate_correlation()
absolute_error, relative_error = all_tasks.calculate_error()
result_list.append([
    network_name, rate, length, efficiency, stretch, score, 
    pearson_correlation, spearman_correlation, absolute_error, relative_error])

df = pd.DataFrame(result_list, columns=[
    'network_name', 'gr_rate', 'gr_length', 'gr_efficiency', 'gr_stretch', 'gr_score', 
    'pearson_correlation', 'spearman_correlation', 'absolute_error', 'relative_error'
    ])

df.to_csv(out_path, index=None)
