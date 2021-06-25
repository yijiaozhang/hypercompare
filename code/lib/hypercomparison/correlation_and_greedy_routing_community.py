"""
Tasks
"""
import numpy as np
import networkx as nx
import random
import scipy.spatial
import scipy.stats
import itertools

import hypercomparison.utils
import hypercomparison.networks

logger = hypercomparison.utils.get_logger(__name__)


class AllTasks:

    """
    Class to 
    1. get the success rate, average path length, efficiency, stretch and GR score
    of greedy routing on an embedding space.

    2. calculate the correlation (Pearson and Spearman correlation coefficient) 
    and error (absolute and relative error) between embedding
    distance and shortest path length in the orignal network.
    If the number of node pairs is bigger than n_samples, only
    n_samples samples will be evaluated for efficiency.

    Input:
        network::hypercomparison.networks.NetworkBase: the network
        embeddings::dict: {node: vector} dictionary of the node vectors
        shortest_path_distance_dict::dict: {(supernode1, supernode2): float} pairwise weighted shortest path length distance of supernetwork
        beta::float: distance tunable parameter
        """

    def __init__(self, network, embeddings, superG, beta):
        self.network = network
        self.embeddings = embeddings
        self.superG = superG
        self.beta = beta
        self.n_nodes = len(self.network.G)
        self.network.index_nodes()
        logger.info("shortest path length of supernetwork start:")
        if len(superG.nodes) < 30000:
            self.superG_shortest_path_length = dict(nx.shortest_path_length(superG, weight='weight'))
        else: 
            self.superG_shortest_path_length = np.zeros((len(superG.nodes), len(superG.nodes)))
            superG_nodes_list = list(superG.nodes())
            temp = 0
            for node in superG_nodes_list:
                temp_distance = dict(nx.shortest_path_length(superG, node, weight='weight'))
                for i in superG_nodes_list:
                    self.superG_shortest_path_length[node][i] = temp_distance[i]
                temp += 1
                if temp % 10000 == 0:
                    logger.info("shortest path length of supernetwork progress: {}/{}".format(temp, len(superG_nodes_list)))
        logger.info("shortest path length of supernetwork complete!")
        if(self.n_nodes < 10000):
            self.distance_matrix = self._get_distance_matrix()
            logger.info("distance matrix complete!")
    
    def _get_distance_matrix(self):
        logger.info("calculate distance matrix")
        self.distance_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for nodeid1 in range(self.n_nodes):
            for nodeid2 in range(self.n_nodes):
                if nodeid1 != nodeid2:
                    node1 = self.network.id2node[nodeid1]
                    node2 = self.network.id2node[nodeid2]
                    l = self.beta*self.superG_shortest_path_length[
                        self.embeddings[node1][1]][self.embeddings[node2][1]
                        ] - (1 - self.beta)*np.log(self.embeddings[node1][0])
                    self.distance_matrix[nodeid1][nodeid2] = l
        return self.distance_matrix

    def _distance_for_large_network(self, node_x, node_y):
        if not hasattr(self, 'superG_shortest_path_length'):
            shortest_path_length = nx.shortest_path_length(self.superG, self.embeddings[node_x][1], self.embeddings[node_y][1], weight='weight')
        else:
            shortest_path_length = self.superG_shortest_path_length[
            self.embeddings[node_x][1]][self.embeddings[node_y][1]
            ]
        result = self.beta*shortest_path_length - (1 - self.beta)*np.log(self.embeddings[node_x][0])
        return result

    def _one_time_routing(self, source, target):
        visited_nodes = []
        visited_nodes.append(source)
        step = 0
        success = 0
        back = source
        while(target not in self.network.G[source]):
            neighbor_distance = []
            for i in self.network.G[source]:
                if(i != back):
                    if hasattr(self, 'distance_matrix'):
                        neighbor_distance.append([(self.distance_matrix[self.network.node2id[i]][self.network.node2id[target]] + self.distance_matrix[self.network.node2id[target]][self.network.node2id[i]])/2, i])
                    else:
                        temp_distance = (self._distance_for_large_network(i, target) + self._distance_for_large_network(target, i))
                        neighbor_distance.append([temp_distance, i])
            neighbor_distance = np.array(neighbor_distance)
            if(len(neighbor_distance) == 0):
                break
            neighbor_distance = random.choice(neighbor_distance[neighbor_distance[:, 0].astype(np.float64) == min(neighbor_distance[:, 0].astype(np.float64))])
            if(neighbor_distance[1] in visited_nodes):
                break
            else:
                back = source
                source = neighbor_distance[1]
                step += 1
                visited_nodes.append(source)
        if(target in self.network.G[source]):
            success = 1
            step += 1
        return step, success

    def greedy_routing(self, number_of_routing = 10000):
        self.network.index_nodes()
        step_list = []
        reverse_step_list = []
        stretch_list = []
        gr_score_list = []
        for i in range(number_of_routing):
            source = random.choice(list(self.network.G.nodes))
            target = random.choice(list(self.network.G.nodes))
            while(target == source):
                target = random.choice(list(self.network.G.nodes))
            step, success = self._one_time_routing(source, target)
            temp_shortest_path_length = self._get_shortest_path_length(self.network.node2id[source], self.network.node2id[target])
            if(success == 1):
                step_list.append(step)
                reverse_step_list.append(1/step)
                stretch_list.append(step/temp_shortest_path_length)
                gr_score_list.append(temp_shortest_path_length/step)
            else:
                gr_score_list.append(0)
        rate = len(step_list)/number_of_routing
        return rate, np.mean(step_list), np.mean(reverse_step_list)*rate, np.mean(stretch_list), np.mean(gr_score_list)

    def calculate_correlation(self):
        """
        Method to calculate the correlation.

        """
        self.network.index_nodes()
        self._calculate_dist()
        pearson_correlation, pearson_pvalue = scipy.stats.pearsonr(self.dist[:,0], self.dist[:,1])
        spearman_correlation, spearman_pvalue = scipy.stats.spearmanr(self.dist[:,0], self.dist[:,1])
        return pearson_correlation, pearson_pvalue, spearman_correlation, spearman_pvalue

    def calculate_error(self):
        """
        Method to calculate raletive error and absolute error 
        between the distance in original network and embedding space

        """
        self.network.index_nodes()
        self._calculate_dist()
        _, relative_error = self._relative_error()
        _, absolute_error = self._absolute_error()

        return absolute_error, relative_error

    def _relative_error(self):
        gamma = 1
        h = 0.1
        error_old = np.mean(np.abs(self.dist[:,1] - gamma*self.dist[:,0])/self.dist[:,1])
        gamma = gamma - h
        error_new = np.mean(np.abs(self.dist[:,1] - gamma*self.dist[:,0])/self.dist[:,1])
        while(np.abs(error_new - error_old) > 0.0001):
            if(error_old - error_new > 0):
                gamma = gamma - h
            else:
                h = -h/2
                gamma = gamma - h
            error_old = error_new
            error_new = np.mean(np.abs(self.dist[:,1]-gamma*self.dist[:,0])/self.dist[:,1])
        
        return gamma, error_new

    def _absolute_error(self):
        gamma = 1
        h = 0.1
        error_old = np.mean(np.abs(self.dist[:,1] - gamma*self.dist[:,0]))
        gamma = gamma - h
        error_new = np.mean(np.abs(self.dist[:,1] - gamma*self.dist[:,0]))
        while(np.abs(error_new - error_old) > 0.0001):
            if(error_old - error_new > 0):
                gamma = gamma - h
            else:
                h = -h/2
                gamma = gamma - h
            error_old = error_new
            error_new = np.mean(np.abs(self.dist[:,1]-gamma*self.dist[:,0]))
        return gamma, error_new

    def _get_sampled_pairs(self, force_regenerate=False, n_samples = 100000):
        self.n_samples = n_samples
        if not hasattr(self, 'sampled_pairs') or force_regenerate:
            if self.n_nodes*(self.n_nodes - 1)/2 < self.n_samples:
                self.sampled_pairs = list(
                    itertools.combinations(range(self.n_nodes), 2)
                )
            else:
                self.sampled_pairs = self._sample_pairs()

    def _sample_pairs(self):
        pair_samples = set()
        while len(pair_samples) < self.n_samples:
            x = random.randint(0, self.n_nodes-1)
            y = random.randint(0, self.n_nodes-1)
            if x == y:
                continue
            # make sure y is bigger than x
            if x > y:
                x, y = y, x
            pair_samples.add((x, y))
        return pair_samples

    def _get_shortest_path_length(self, nodeid1, nodeid2):
        if hasattr(self.network, 'shortest_path_length_matrix'):
            return self.network.shortest_path_length_matrix[nodeid1, nodeid2]
        else:
            return nx.shortest_path_length(
                self.network.G,
                self.network.id2node[nodeid1],
                self.network.id2node[nodeid2]
            )
  
    def _calculate_dist(self, force_regenerate=False):
        if not hasattr(self, 'dist') or force_regenerate:
            self._get_sampled_pairs()
            self.dist = []
            if hasattr(self, 'distance_matrix'):
                logger.info("distance matrix exists.")
                for nodeid1, nodeid2 in self.sampled_pairs:
                    embedding_distance = (self.distance_matrix[nodeid1][nodeid2] + self.distance_matrix[nodeid1][nodeid2])/2
                    if np.isfinite(embedding_distance):
                        path_distance = self._get_shortest_path_length(nodeid1, nodeid2)
                        self.dist.append([embedding_distance, path_distance])
            else:
                for nodeid1, nodeid2 in self.sampled_pairs:
                    node1 = self.network.id2node[nodeid1]
                    node2 = self.network.id2node[nodeid2]
                    l1 = self.beta*self.superG_shortest_path_length[
                        self.embeddings[node1][1]][self.embeddings[node2][1]
                        ] - (1 - self.beta)*np.log(self.embeddings[node1][0])
                    l2 = self.beta*self.superG_shortest_path_length[
                        self.embeddings[node1][1]][self.embeddings[node2][1]
                        ] - (1 - self.beta)*np.log(self.embeddings[node1][0])
                    embedding_distance = (l1+l2)/2
                    if np.isfinite(embedding_distance):
                        path_distance = self._get_shortest_path_length(nodeid1, nodeid2)
                        self.dist.append([embedding_distance, path_distance])
            self.dist = np.array(self.dist)
