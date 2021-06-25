import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

import hypercomparison.utils
import hypercomparison.networks
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
        proximity_function='dot'
    ):
        """
        :param test_edges: list of test edges
        :param negative_edges: list of negative edges
        :param emb: embedding results
        :param name: name of model for record
        """
        self.test_edges = test_edges
        self.negative_edges = negative_edges
        self.emb = emb
        self.name = name

        self.link_prediction_score_positive = []
        self.link_prediction_score_negative = []

        self.proximity_function = proximity_function
        if self.proximity_function not in [
            'dot', 'cos', 'euclidean_distance', 'hyperbolic_distance',
            'poincare_distance']:
            raise NotImplementedError

    def do_link_prediction(self):
        """
        Execute link prediction
        """
        self.calculate_link_prediction_score()
        roc_auc_value, aupr_value, average_precision, precision_score = self.calculate_different_metrics()
        logger.info(self.name)
        logger.info(roc_auc_value)
        return roc_auc_value, aupr_value, average_precision, precision_score

    def _calculate_hyperbolic_distance(self, node_x, node_y):
        delta_theta = np.pi - np.abs(np.pi - np.abs(self.emb[node_x][0] - self.emb[node_y][0]))
        temp = np.cosh(self.emb[node_x][1])*np.cosh(self.emb[node_y][1]) - np.sinh(self.emb[node_x][1])*np.sinh(self.emb[node_y][1])*np.cos(delta_theta)
        if temp < 1:
            temp = 1
        return np.arccosh(temp)
    
    def _calculate_poincare_distance(self, node_x, node_y):
        temp = np.linalg.norm(np.array(self.emb[node_x]) - np.array(self.emb[node_y]))**2 / ((1 - np.linalg.norm(self.emb[node_x])**2) * (1 - np.linalg.norm(self.emb[node_y])**2))
        temp = 1 + 2*temp
        if temp < 1:
            temp = 1
        return np.arccosh(temp)

    def calculate_link_prediction_score(self):
        """
        Calculate similarity score for test and negative edges
        """
        self.link_prediction_score_positive = np.array(
            self.calculate_score(self.test_edges)
        )
        self.link_prediction_score_negative = np.array(
            self.calculate_score(self.negative_edges)
        )

    def calculate_score(self, edge_list):
        """
        Calculate similarity score for edge_list
        :param edge_list: list of target edges.
        :return: score_list: score list of given edge_lists
        """
        embs = np.array(
            [[self.emb[str(source)], self.emb[str(target)]] for source, target in edge_list]
        )

        if self.proximity_function == 'dot':
            score_list = [np.dot(source_emb, target_emb) for source_emb, target_emb in embs]
        elif self.proximity_function == 'cos':
            score_list = cosine_similarity(embs[:, 0], embs[:, 1])
            score_list = score_list.diagonal()
        elif self.proximity_function == 'euclidean_distance':
            score_list = [-1*np.linalg.norm(source_emb - target_emb) for source_emb, target_emb in embs]
        elif self.proximity_function == 'hyperbolic_distance':
            score_list = [-1*self._calculate_hyperbolic_distance(source, target) for source, target in edge_list]
        elif self.proximity_function == 'poincare_distance':
            score_list = [-1*self._calculate_poincare_distance(source, target) for source, target in edge_list]
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