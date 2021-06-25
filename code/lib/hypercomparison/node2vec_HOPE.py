"""
Network embedding related functions and algorithms.
"""
import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import random
import time
import scipy.sparse as sp
from scipy import linalg

from sklearn import preprocessing

import hypercomparison.utils
import hypercomparison.networks
logger = hypercomparison.utils.get_logger(__name__)

########################################
########################################
# Node2vec
########################################
class Node2vec:
    def __init__(
        self,
        dimension=128,
        walk_length=80,
        walk_num=10,
        window_size=10,
        worker=1,
        iteration=1,
        p=1,
        q=1
    ):
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration
        self.p = p
        self.q = q

    def train(self, G):
        self.G = G
        is_directed = nx.is_directed(self.G)
        for i, j in G.edges():
            G[i][j]["weight"] = G[i][j].get("weight", 1.0)
            if not is_directed:
                G[j][i]["weight"] = G[j][i].get("weight", 1.0)
        self._preprocess_transition_probs()
        walks = self._simulate_walks(self.walk_num, self.walk_length)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(
            walks,
            size=self.dimension,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.worker,
            iter=self.iteration,
        )
        self.id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        #temp_embeddings = np.array([list(model.wv[str(self.id2node[i])]) for i in range(len(self.id2node))]) # centering the coordinates
        #center_point = temp_embeddings.mean(axis=0) #
        #temp_embeddings -= center_point #
        #self.embeddings = {str(self.id2node[i]): temp_embeddings[i] for i in range(len(self.id2node))}
        self.embeddings = {
            str(self.id2node[i]): model.wv[str(self.id2node[i])] for i in range(len(self.id2node))
        }
        return self.embeddings

    def _node2vec_walk(self, walk_length, start_node):
        # Simulate a random walk starting from start node.
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[hypercomparison.utils.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    next = cur_nbrs[
                        hypercomparison.utils.alias_draw(
                            alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1]
                        )
                    ]
                    walk.append(next)
            else:
                break

        return walk

    def _simulate_walks(self, num_walks, walk_length):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        logger.info("Walk iteration:")
        for walk_iter in range(num_walks):
            if walk_iter % 10 == 0:
                logger.info(str(walk_iter + 1) + "/" + str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(
                    self._node2vec_walk(walk_length=walk_length, start_node=node)
                )

        return walks

    def _get_alias_edge(self, src, dst):
        # Get the alias edge setup lists for a given edge.
        G = self.G
        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / self.p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return hypercomparison.utils.alias_setup(normalized_probs)

    def _preprocess_transition_probs(self):
        # Preprocessing of transition probabilities for guiding the random walks.
        G = self.G
        is_directed = nx.is_directed(self.G)

        logger.info(len(list(G.nodes())))
        logger.info(len(list(G.edges())))

        s = time.time()
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = hypercomparison.utils.alias_setup(normalized_probs)

        t = time.time()
        logger.info("alias_nodes {}".format(t - s))

        alias_edges = {}
        s = time.time()

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])

        t = time.time()
        logger.info("alias_edges {}".format(t - s))

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    def to_data_frame(self):
        embedding_df = pd.DataFrame(self.embeddings)
        node_id_df = pd.DataFrame(list(self.id2node.items()), columns=['index', 'node_id'])
        self.embedding_df = node_id_df.merge(embedding_df, left_on='index', right_index=True)


########################################
########################################
# HOPE
########################################
class HOPE:
    """
    Ou, Mingdong, et al. Asymmetric transitivity preserving graph embedding.
    Proceedings of the 22nd ACM SIGKDD international conference on Knowledge
    discovery and data mining. ACM, 2016.

    Implementation borrowed from https://github.com/THUDM/cogdl/blob/master/cogdl/models/emb/hope.py
    with modification
    This implementation use Katz similarity of the nodes, which is claimed to
    be the best in the paper

    HOPE use the numpy linear algebra lib, which by default uses multiple threads.
    To disable it, export OPENBLAS_NUM_THREADS=1 or MKL_NUM_THREADS=1
    in shell depending on the backend of the numpy installation.
    """
    def __init__(self, dimension=128, beta=0.01):
        self.dimension = dimension
        self.beta = beta

    def train(self, G):
        self.G = G
        self.id2node = dict(zip(range(len(G)), G))

        adj = nx.adjacency_matrix(self.G).todense()
        n = adj.shape[0]
        # The author claim that Katz has superior performance in related tasks
        # S_katz = (M_g)^-1 * M_l = (I - beta*A)^-1 * beta*A = (I - beta*A)^-1 * (I - (I -beta*A))
        #        = (I - beta*A)^-1 - I
        katz_matrix = np.asarray((np.eye(n) - self.beta * np.mat(adj)).I - np.eye(n))
        self.embeddings_matrix = self._get_embedding(katz_matrix, self.dimension)
        #center_point = self.embeddings_matrix.mean(axis=0) # centering the coordinates
        #self.embeddings_matrix -= center_point # centering the coordinates
        self.embeddings = {
            str(self.id2node[i]): self.embeddings_matrix[i] for i in range(len(self.id2node))
        }

        return self.embeddings

    def _get_embedding(self, matrix, dimension):
        # get embedding from svd and process normalization for ut and vt
        ut, s, vt = sp.linalg.svds(matrix, int(dimension / 2))
        emb_matrix_1, emb_matrix_2 = ut, vt.transpose()

        emb_matrix_1 = emb_matrix_1 * np.sqrt(s)
        emb_matrix_2 = emb_matrix_2 * np.sqrt(s)
        emb_matrix_1 = preprocessing.normalize(emb_matrix_1, "l2")
        emb_matrix_2 = preprocessing.normalize(emb_matrix_2, "l2")
        features = np.hstack((emb_matrix_1, emb_matrix_2))
        return features