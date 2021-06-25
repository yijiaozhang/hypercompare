"""
Implementations of Isomap
Metric MDS + shortest path length matrix of a network
"""
from sklearn import manifold
import hypercomparison.utils
import hypercomparison.networks
logger = hypercomparison.utils.get_logger(__name__)


class MDSBase:
    def __init__(self, dimension):
        self.dimension = dimension

    def train(self, network):
        self.network = network
        self.network.generate_shortest_path_length_matrix()
        self.embeddings_matrix = self._get_embedding()

        if len(self.embeddings_matrix) == 0:
            self.embeddings = {}
        else:
            self.embeddings = {
                str(self.network.id2node[i]): self.embeddings_matrix[i] for i in range(len(self.network.id2node))
            }
        return self.embeddings


class Isomap(MDSBase):
    def __init__(self, dimension=2):
        super(Isomap, self).__init__(dimension)

    def _get_embedding(self):
        return manifold.MDS(
            self.dimension, dissimilarity="precomputed").fit_transform(
                self.network.shortest_path_length_matrix)
