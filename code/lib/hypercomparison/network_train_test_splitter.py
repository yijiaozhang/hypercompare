import logging
import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class NetworkTrainTestSplitterWithMST():
    def __init__(self, G, fraction=0.3):
        """
        Only support undirected Network
        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        self.G = G
        self.original_edge_set = set(G.edges)
        self.node_list = list(G.nodes)
        self.total_number_of_edges = len(G.edges)
        self.number_of_test_edges = int(self.total_number_of_edges * fraction)

        self.test_edges = []
        self.negative_edges = []
        for x, y in G.edges:
            self.original_edge_set.add((y, x))
        

    def train_test_split(self):
        """
        Split train and test edges with MST.
        Train network should have a one weakly connected component.
        """
        logging.info("Initiate train test set split with MST")

        MST = nx.minimum_spanning_tree(self.G)
        remained_edge_set = np.array(list(set(self.G.edges()) - set(MST.edges)))
        candidate_idxs = np.random.choice(
            range(len(remained_edge_set)), self.number_of_test_edges, replace=False
        )
        self.test_edges = remained_edge_set[candidate_idxs]
        self.G.remove_edges_from(self.test_edges)
        return self.G, self.test_edges

    def generate_negative_edges(self):
        """
        Generate a negative samples for link prediction task
        """
        logging.info("Initiate generating negative edges")
        while len(self.negative_edges) != self.number_of_test_edges:
            #logging.info("generating negative edges, {}/{}".format(len(self.negative_edges), self.number_of_test_edges))
            node_x = np.random.randint(0, len(self.node_list), (self.number_of_test_edges - len(self.negative_edges)))
            node_y = np.random.randint(0, len(self.node_list), (self.number_of_test_edges - len(self.negative_edges)))

            node_x_unique = node_x[node_x > node_y]
            node_y_unique = node_y[node_x > node_y]

            temp_list = [(self.node_list[node_x_unique[i]], self.node_list[node_y_unique[i]]) for i in range(len(node_x_unique))]
            temp_set = set(temp_list)

            duplicate_set1 = temp_set & self.original_edge_set
            duplicate_set2 = temp_set & set(self.negative_edges)
            temp_list = list(temp_set - duplicate_set1 - duplicate_set2)
            self.negative_edges.extend(temp_list)  
        return self.negative_edges
