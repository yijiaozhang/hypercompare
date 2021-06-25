# -*- coding: utf8 -*-
"""
Classes to handle network datasets
"""
import hypercomparison.utils
import os
import networkx as nx
import numpy as np
import math

logger = hypercomparison.utils.get_logger(__name__)

package_directory = os.path.dirname(os.path.abspath(__file__))
package_root_directory = os.path.dirname(package_directory)

local_dataset_directory_real = os.path.join(package_directory, 'data')


########################################
########################################
# Base class
########################################
class NetworkBase:
    """
    Base class for networks, provides some basic operations
    """
    def __init__(self):
        pass

    def get_largest_component(self, directed=False):
        """
        Return the largest connected component subgraph for undirected networks
        Return the largest weakly connected component subgraph for directed
        networks
        """
        if directed:
            #return max(nx.weakly_connected_component_subgraphs(self.G), key=len) #for networkx version older than 2.1
            return self.G.subgraph(max(nx.weakly_connected_components(self.G), key=len)).copy()
        else:
            #return max(nx.connected_component_subgraphs(self.G), key=len)
            return self.G.subgraph(max(nx.connected_components(self.G), key=len)).copy()

    def convert_to_undirected(self):
        """
        Convert the graph to undirected.
        """
        self.G = self.G.to_undirected()

    def remove_selfloops(self):
        """
        Remove selfloops edges.
        """
        self.G.remove_edges_from(self.G.selfloop_edges())

    def remove_parallel_edges(self):
        """
        Remove parallel edges.
        """
        self.G = nx.Graph(self.G)

    def generate_shortest_path_length_matrix(self):
        if hasattr(self, 'shortest_path_length_matrix'):
            logger.info("Shortest path length exists.")
            return

        number_of_nodes = len(self.G.nodes)
        self.index_nodes()

        self.shortest_path_length_matrix = np.zeros((
            number_of_nodes, number_of_nodes
        ))
        shortest_path_length_dict = dict(nx.shortest_path_length(self.G))

        for nodeid1 in range(number_of_nodes):
            for nodeid2 in range(number_of_nodes):
                l = shortest_path_length_dict[
                    self.id2node[nodeid1]
                ][
                    self.id2node[nodeid2]
                ]
                self.shortest_path_length_matrix[nodeid1][nodeid2] = l

    def index_nodes(self):
        if not hasattr(self, 'id2node'):
            self.id2node = dict(enumerate(self.G.nodes))

        if not hasattr(self, 'node2id'):
            self.node2id = {value: key for key, value in self.id2node.items()}

    def dump_network(self, path):
        nx.write_edgelist(self.G, path)

########################################
# Reloaded networks
class ReloadedNetwork(NetworkBase):
    def __init__(self, data_path):
        super(ReloadedNetwork, self).__init__()
        self.data_path = data_path
        self._load_network()
    
    def _load_network(self):
        self.G = nx.read_edgelist(self.data_path, create_using=nx.Graph())
        #self.G = max(nx.connected_component_subgraphs(self.G), key=len)
        self.G = self.G.subgraph(max(nx.connected_components(self.G), key=len)).copy()


########################################
########################################
# Real networks
########################################
class RealNetwork(NetworkBase):
    """
    Base class to load real networks
    """   
    def __init__(self, network_name):
        """
        Input:
            network_name::string: network name
        """
        super(RealNetwork, self).__init__()
        self.network_name = network_name
        self.data_path = os.path.join(local_dataset_directory_real, '%s_edges.txt'%self.network_name)
        self._load_network()
    
    def _load_network(self):
        if not os.path.exists(self.data_path):
            logger.error("Data file does not exist.")
        else:
            self.G = nx.read_edgelist(self.data_path, create_using=nx.Graph())
            #self.G = max(nx.connected_component_subgraphs(self.G), key=len)
            self.G = self.G.subgraph(max(nx.connected_components(self.G), key=len)).copy()
            

    def __repr__(self):
        if hasattr(self, 'G'):
            representation = [
                '[Name]:        {}\n'.format(self.network_name),
                '[Nodes]:       {}\n'.format(len(self.G.nodes)),
                '[Edges]:       {}\n'.format(len(self.G.edges))
            ]
            return ''.join(representation)
        else:
            return 'Failed to download the network, please manually download {} edges to {}.'.format(
                self.network_name,
                local_dataset_directory_real
            )


########################################
########################################
# Network models
########################################
class ModelNetwork(NetworkBase):
    """
    Base class to generate and handle network models.
    It uses configuration model to generate the network.
    There are self loops and parallel as side products during the
    generation, therefore the final network might not have
    the exact degree sequence as expected.

    Assuming undirected graphs.
    """
    def __init__(self):
        super(ModelNetwork, self).__init__()
        self.regenerate_network()

    def _generate_degree_sequence(self):
        raise NotImplementedError

    def regenerate_network(self, ensure_connected=True, regenerate_sequence=False):
        if not hasattr(self, 'degree_sequence') or regenerate_sequence:
            self._generate_degree_sequence()

        G = nx.configuration_model(self.degree_sequence, create_using=nx.Graph())
        if ensure_connected:
            logger.info("Ensuring connected graph in configuration model..")
            count = 1
            while not nx.is_connected(G):
                logger.info("Attempt {}..".format(count))
                G = nx.configuration_model(self.degree_sequence, create_using=nx.Graph())
                count += 1
                if count > 30:
                    #G = max(nx.connected_component_subgraphs(G), key=len)
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        self.G = G
        self.remove_selfloops()

    def __repr__(self):
        if hasattr(self, 'G'):
            representation = [
                '[Name]:        {}\n'.format(self.data_meta_info['network_name']),
                '[Nodes]:       {}\n'.format(len(self.G.nodes)),
                '[Edges]:       {}\n'.format(len(self.G.edges)),
            ]
        return ''.join(representation)

class PoissonNetwork(ModelNetwork):
    """
    Class to handle Poisson degree distribution networks

    Input:
        N::int: number of nodes in the network
        average_degree::int: average degree of the nodes
    """
    data_meta_info = {
        'network_name': 'Poisson Network'
    }
    def __init__(self, N, average_degree):
        self.N = N
        self.average_degree = average_degree
        super(PoissonNetwork, self).__init__()

    def _generate_degree_sequence(self):
        self.degree_sequence = []
        total_prob = 0
        num_i = []
        temp = 0
        for i in range(1, 3 * self.average_degree + 1):
            total_prob += math.pow(self.average_degree, i) * math.exp(-self.average_degree) / math.factorial(i)

        for i in range(1, 3 * self.average_degree + 1):
            num = int(round(self.N * math.pow(self.average_degree, i) * math.exp(-self.average_degree) / (math.factorial(i) * total_prob)))
            num_i.append(num)
            temp += num

        for i in range(3 * self.average_degree):
            if(num_i[i] > 0):
                num_i[i] += self.N - temp
                break

        for i in range(1, 3 * self.average_degree + 1):
            for j in range(num_i[i-1]):
                self.degree_sequence.append(i)

        if(sum(self.degree_sequence)%2 == 1):
            self.degree_sequence[0] += 1


class PowerlawNetwork(ModelNetwork):
    """
    Class to handle power law degree distribution networks

    Input:
        N::int: number of nodes in the network
        gamma::float: exponent
        k_min::int: lower bound of degree
        k_max::int: upper bound of degree
    """
    data_meta_info = {
        'network_name': 'Power law Network'
    }
    def __init__(self, N, gamma, k_min, k_max):
        self.N = N
        self.gamma = gamma
        self.k_min = k_min
        self.k_max = k_max
        super(PowerlawNetwork, self).__init__()

    def _generate_degree_sequence(self):
        self.degree_sequence = []
        total_prob = 0
        num_i = []
        temp = 0
        for i in range(self.k_min, self.k_max+1):
            total_prob += math.pow(i, -self.gamma)
        total_prob = 1 / total_prob

        for i in range(self.k_min, self.k_max+1):
            num = int(round(self.N * math.pow(i, -self.gamma) * total_prob))
            num_i.append(num)
            temp += num

        for i in range(0, self.k_max - self.k_min +1):
            if(num_i[i] > 0):
                num_i[i] += self.N - temp
                break

        for i in range(self.k_min, self.k_max+1):
            for j in range(num_i[i - self.k_min]):
                self.degree_sequence.append(i)
        if(sum(self.degree_sequence)%2 == 1):
            self.degree_sequence[0] += 1
        return self.degree_sequence
