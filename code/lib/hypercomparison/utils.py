"""
Various utilities
"""
import os
import numpy as np
import networkx as nx
import scipy.stats
import scipy.spatial
import errno
import urllib.request
import logging

def get_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)
    # Create handlers
    handler = logging.StreamHandler()
    # Create formatters and add it to handlers
    logger_format = logging.Formatter(
        "%(asctime)s@%(name)s:%(levelname)s: %(message)s")
    handler.setFormatter(logger_format)
    # Add handlers to the logger
    logger.addHandler(handler)
    # Set level
    level = logging.getLevelName("INFO")
    logger.setLevel(level)
    return logger

logger = get_logger(__name__)


########################################
########################################
# Paths related
########################################
def makedirs(path):
    """
    Create a new directory on the disk

    Input:
        path::string:  target directory
    """
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e


def download_url(url, folder, log=True, file_name=None):
    """Downloads the content of an URL to a folder.

    Input:
        url::string:      target url
        folder::string:   target folder
        log::bool:        weather to log the information
        file_name:string: file_name for the target file

    Retrun:
        path::string: the path of the content
    """

    if file_name is None:
        file_name = url.rpartition('/')[2]
    path = os.path.join(folder, file_name)

    if os.path.exists(path):  # pragma: no cover
        if log:
            logger.info('File exsits: {}'.format(file_name))
        return path

    if log:
        logger.info('Downloading: {}'.format(url))

    makedirs(folder)

    # Faking the request headers to bypass some download restrictions
    hdr = { 'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)' }
    req = urllib.request.Request(url, headers=hdr)
    data = urllib.request.urlopen(req)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


########################################
########################################
# Network IO
#    Implement a standard way to dump and load
#    Networks
########################################
def dump_network(G, path):
    """
    Dump the network as edgelist
    """
    nx.write_edgelist(G, path)


def load_network(path, weighted=False, directed=False):
    """
    Load the edgelist as a networkx.Graph() object.

    Input:
        path::string: path to the edgelist
        weighted::bool: whether the network has edge weights
        directed::bool: whether the network is directed

    Return:
        ::networkx.Graph: the loaded network
    """
    data = (("weight", float),) if weighted else True
    G = nx.read_edgelist(path, nodetype=str, data=data, create_using=nx.DiGraph())

    if not weighted:
        for source, target in G.edges():
            G[source][target]['weight'] = 1

    if not directed:
        G = G.to_undirected()
    return G


########################################
########################################
# Alias sampling method
########################################
# Function implemenentation for node2vec etc
def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to http://cgi.cs.mcgill.ca/~enewel3/posts/alias-method/index.html
    for details

    Input:
        probs::list: a list of probabilites
                     ensure sum(probs) == 1

    Return:
        J::numpy.ndarray: alias lookup table
        q::numpy.ndarray: threshold table
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.

    Input:
        J::numpy.ndarray: alias lookup table
        q::numpy.ndarray: threshold table

    Return:
        ::Int: an index randomly selected according to the probability list
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


# Class implemenentation for general purpose
class AliasSampler:
    """
    O(1) descrete distribution sampling with the alias method.
    Refer to http://cgi.cs.mcgill.ca/~enewel3/posts/alias-method/index.html
    for mechanism details.

    Usage:
        alias_sampler = AliasSampler(probs)
        random_index_1 = alias_sampler.draw()
        random_index_2 = alias_sampler.draw()
    """
    def __init__(self, probs):
        """
        Input:
            probs::list: list of numbers
                         the class will normalize the probabilites automatically
        """
        self.probs = probs
        self._normalize_prob()
        self._setup_alias()

    def _normalize_prob(self):
        """
        Method to normalize the probability list
        """
        normalizer = sum(self.probs)
        normalized_probs = [float(prob) / normalizer for prob in self.probs]
        self.probs = normalized_probs

    def _setup_alias(self):
        """
        Setup the alias lookup table and threshold table
        """
        K = len(self.probs)
        self.q = np.zeros(K)
        self.J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(self.probs):
            self.q[kk] = K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.q[large] = self.q[large] + self.q[small] - 1.0
            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw(self):
        """
        Generate a random index
        """
        K = len(self.J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < self.q[kk]:
            return kk
        else:
            return self.J[kk]

    def draw_n(self, n):
        """
        Generate n random indexes

        Input:
            n::int: number of random indexes to generate

        Return:
            ::list: a list of random indexes
                    with repetation
        """
        return [self.draw() for i in range(n)]



########################################
########################################
# Stats related
########################################
def mean_confidence_interval(data, confidence=0.95):
    """
    Function to calculate the mean valuea and confidence interval
    for a given array of data
    Copied from https://stackoverflow.com/a/15034143

    Input:
        data::list: list of data

    Return:
        m::float: mean value
        m-h::float: lower confidence
        m+h::float: upper confidence
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


########################################
########################################
# Metrics
########################################
def cosine_similarity(x, y):
    """
    Calculate the cosine similarity of x and y: <x, y> / (|x| * |y|),
    if x and y are 1-D vectors

    Calculate the pairwise cosine similarity for x and y,
    if x and y are m*d and n*d matrix where d is the dimension

    Input:
        x::numpy.ndarray: vector/matrix x
        y::numpy.ndarray: vector/matri y

    Return:
        ::float: the cosine similarity between x and y
    """
    if len(x.shape) == 1 and len(y.shape) == 1:
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return 1 - scipy.spatial.distance.cdist(x, y, 'cosine')
