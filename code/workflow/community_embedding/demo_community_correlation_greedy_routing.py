import sys
import hypercomparison.utils
import hypercomparison.networks
import hypercomparison.community_embedding
import hypercomparison.correlation_and_greedy_routing_community
import numpy as np
import pandas as pd
import networkx as nx

logger = hypercomparison.utils.get_logger(__name__)

network_name = sys.argv[1]
beta = float(sys.argv[2])
out_path = sys.argv[-1]

result_list = []
network = hypercomparison.networks.RealNetwork(network_name)

embeddings, superG = hypercomparison.community_embedding.Comm_Infomap().train(network) # community embedding based on Infomap algorithm
#embeddings, superG = hypercomparison.community_embedding.Comm_Louvain().train(network) # community embedding based on Louvain algorithm

logger.info("Working on network {}, {}".format(network_name, beta))
all_tasks = hypercomparison.correlation_and_greedy_routing_community.AllTasks(network, embeddings, superG, beta)
rate, length, efficiency, stretch, score = all_tasks.greedy_routing()
pearson_correlation, _, spearman_correlation, _ = all_tasks.calculate_correlation()
absolute_error, relative_error = all_tasks.calculate_error()
result_list.append([
    network_name, beta, rate, length, efficiency, stretch, score, 
    pearson_correlation, spearman_correlation, absolute_error, relative_error])

df = pd.DataFrame(result_list, columns=[
    'network_name', 'beta', 'gr_rate', 'gr_length', 'gr_efficiency', 'gr_stretch', 'gr_score', 
    'pearson_correlation', 'spearman_correlation', 'absolute_error', 'relative_error'
    ])

df.to_csv(out_path, index=None)
