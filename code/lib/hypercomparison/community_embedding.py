"""
Implementation of community embedding based on Infomap and Louvain community detection algorithm
"""
import numpy as np
import networkx as nx
from infomap import Infomap
import community as community_louvain
from numpy import ma
import hypercomparison.utils
import hypercomparison.networks
logger = hypercomparison.utils.get_logger(__name__)


########################################
########################################
# Community embedding
########################################
class CommunityBase:
    """
    Faqeeh, Ali, Saeed Osat, and Filippo Radicchi. 
    "Characterizing the analogy between hyperbolic embedding and community structure of complex networks." 
    Physical review letters 121.9 (2018): 098301.
    """
    def __init__(self):
        self

    def train(self, network):
        self.network = network
        self.network.index_nodes()
        self.embeddings, self.superG = self._get_embedding_and_supergraph()
        return self.embeddings, self.superG

class Comm_Infomap(CommunityBase):
    def __init__(self):
        super(Comm_Infomap, self).__init__()

    def _get_embedding_and_supergraph(self):
        #community index start from 1
        im = Infomap("--two-level")
        for e in list(self.network.G.edges):
            im.addLink(int(e[0]), int(e[1]))
        im.run()
        node_type = type(list(self.network.G.nodes)[0])
        community_dict = {}
        for node_id, module_id in im.modules:
            community_dict[node_type(node_id)] = module_id-1
        degree_dict = dict(nx.degree(self.network.G))
        self.embeddings = {node_type(self.network.id2node[i]): [
            degree_dict[node_type(self.network.id2node[i])],
            community_dict[node_type(self.network.id2node[i])]
        ] for i in range(len(self.network.id2node))}
        #logger.info("connecting super graph...")
        #create weighted supernetwork and calculate pairwise distance between supernodes
        supernode_set = set(community_dict.values())
        supernetwork_matrix = np.zeros((len(supernode_set), len(supernode_set)))
        for e in list(self.network.G.edges):
            node1 = community_dict[e[0]]
            node2 = community_dict[e[1]]
            if node1 != node2:
                supernetwork_matrix[node1][node2] +=1
                supernetwork_matrix[node2][node1] +=1
        #logger.info("loop for edges complete...")
        for node1 in supernode_set:
            list_node1 = list(k for k,v in community_dict.items() if v == node1)
            all_length = sum([degree_dict[node] for node in list_node1])
            supernetwork_matrix[node1] = supernetwork_matrix[node1]/all_length
            supernetwork_matrix[node1] = (1-ma.log(supernetwork_matrix[node1])).filled(0)
        #logger.info("weight calculated...")
        nodelist1, nodelist2 = np.where(supernetwork_matrix != 0)
        self.superG = nx.DiGraph((nodelist1[i], nodelist2[i], {'weight': supernetwork_matrix[nodelist1[i]][nodelist2[i]]}) for i in range(len(nodelist1)))
        #logger.info("super graph connected!")
        if len(self.superG.nodes) == 0:
            self.superG = nx.DiGraph()
            self.superG.add_nodes_from(list(supernode_set))
        return self.embeddings, self.superG

class Comm_Louvain(CommunityBase):
    def __init__(self):
        super(Comm_Louvain, self).__init__()

    def _get_embedding_and_supergraph(self):
        #community index start from 0
        partition = community_louvain.best_partition(self.network.G)
        node_type = type(list(self.network.G.nodes)[0])
        degree_dict = dict(nx.degree(self.network.G))
        self.embeddings = {node_type(self.network.id2node[i]): [
            degree_dict[node_type(self.network.id2node[i])],
            partition[node_type(self.network.id2node[i])]
        ] for i in range(len(self.network.id2node))}
        #logger.info("connecting super graph...")
        #create weight supernetwork and calculate pairwise distance 
        supernode_set = set(partition.values())
        supernetwork_matrix = np.zeros((len(supernode_set), len(supernode_set)))
        for e in list(self.network.G.edges):
            node1 = partition[e[0]]
            node2 = partition[e[1]]
            if node1 != node2:
                supernetwork_matrix[node1][node2] +=1
                supernetwork_matrix[node2][node1] +=1
        #logger.info("loop for edges complete...")
        for node1 in supernode_set:
            list_node1 = list(k for k,v in partition.items() if v == node1)
            all_length = sum([degree_dict[node] for node in list_node1])
            supernetwork_matrix[node1] = supernetwork_matrix[node1]/all_length
            supernetwork_matrix[node1] = (1-ma.log(supernetwork_matrix[node1])).filled(0)
        #logger.info("weight calculated...")
        nodelist1, nodelist2 = np.where(supernetwork_matrix != 0)
        self.superG = nx.DiGraph((nodelist1[i], nodelist2[i], {'weight': supernetwork_matrix[nodelist1[i]][nodelist2[i]]}) for i in range(len(nodelist1)))
        if len(self.superG.nodes) == 0:
            self.superG = nx.DiGraph()
            self.superG.add_nodes_from(list(supernode_set))
        #logger.info("super graph connected!")
        return self.embeddings, self.superG