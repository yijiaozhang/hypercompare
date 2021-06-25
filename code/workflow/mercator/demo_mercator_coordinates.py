import sys
import hypercomparison.utils
import hypercomparison.networks
import mercator

logger = hypercomparison.utils.get_logger(__name__)
network_edges_path = sys.argv[1]
network_name = sys.argv[2]
mercator_coordinates_path = sys.argv[3]

network = hypercomparison.networks.RealNetwork(network_name)
mercator.embed(network_edges_path, output_name=mercator_coordinates_path)