import numpy as np
import networkx as nx
from sklearn import metrics

import hypercomparison.utils
logger = hypercomparison.utils.get_logger(__name__)

class LinkPredictionTask(object):
    """
    Link prediction object
    Given a test edges and negative edges and embedding, calculate ROC_AUC values.
    """

    def __init__(
        self,
        test_edges,
        negative_edges,
        emb,
        name,
        superG, 
        beta
    ):
        """
        :param test_edges: list of test edges
        :param negative_edges: list of negative edges
        :param emb: embedding results
        :param name: name of model for record
        :param shortest_path_distance_dict::dict: {(supernode1, supernode2): float} pairwise weighted shortest path length distance of supernetwork
        :param beta::float: distance tunable parameter
        """
        self.test_edges = test_edges
        self.negative_edges = negative_edges
        self.emb = emb
        self.name = name
        self.superG = superG
        self.beta = beta
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
        self.link_prediction_score_positive = []
        self.link_prediction_score_negative = []

    def do_link_prediction(self):
        """
        Execute link prediction
        """
        self.calculate_link_prediction_score()
        roc_auc_value, aupr_value, average_precision, precision_score = self.calculate_different_metrics()
        logger.info(self.name)
        logger.info(roc_auc_value)
        return roc_auc_value, aupr_value, average_precision, precision_score

    def _calculate_community_distance(self, node_x, node_y):
        if not hasattr(self, 'superG_shortest_path_length'):
            shortest_path_length = nx.shortest_path_length(self.superG, self.emb[node_x][1], self.emb[node_y][1], weight='weight')
        else:
            shortest_path_length = self.superG_shortest_path_length[
                self.emb[node_x][1]][self.emb[node_y][1]
                ] 
        result = self.beta*shortest_path_length - (1 - self.beta)*np.log(self.emb[node_x][0])
        return result

    def calculate_link_prediction_score(self):
        """
        Calculate similarity score for test and negative edges
        """
        logger.info("Calculate link prediction score positive")
        self.link_prediction_score_positive = np.array(
            self.calculate_score(self.test_edges)
        )
        logger.info("Calculate link prediction score negative")
        self.link_prediction_score_negative = np.array(
            self.calculate_score(self.negative_edges)
        )

    def calculate_score(self, edge_list):
        """
        Calculate similarity score for edge_list
        :param edge_list: list of target edges.
        :return: score_list: score list of given edge_lists
        """
        #score_list = []
        #for source, target in edge_list:
        #    if len(score_list)%10000 == 0:
        #        logger.info("progress, {}/{}".format(len(score_list), len(edge_list)))
        #    score_list.append(-1*(self._calculate_community_distance(source, target) + self._calculate_community_distance(target, source))/2)    
        score_list = [-1*(self._calculate_community_distance(source, target) + self._calculate_community_distance(target, source))/2 for source, target in edge_list]
        return score_list

    def calculate_different_metrics(self):
        """
        Calculate ROC_AUC values
        Calculate AUPR--area under precision-recall curve
        Calculate Average precision score
        Calculate precision score
        """
        logger.info("Calculate ROC_AUC values")
        y_true = np.concatenate(
            [
                np.ones_like(self.link_prediction_score_positive),
                np.zeros_like(self.link_prediction_score_negative),
            ]
        )
        y_score = np.concatenate(
            [self.link_prediction_score_positive, self.link_prediction_score_negative],
            axis=0,
        )
        y_median = np.median(y_score)
        y_predict = np.where(y_score > y_median, 1, 0)
        roc_auc_value = metrics.roc_auc_score(y_true, y_score)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        aupr_value = metrics.auc(recall, precision)
        average_precision = metrics.average_precision_score(y_true, y_score)
        precision_score = metrics.precision_score(y_true, y_predict)
        return roc_auc_value, aupr_value, average_precision, precision_score