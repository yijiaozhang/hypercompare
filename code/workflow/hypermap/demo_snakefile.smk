"""
Snakefile to get the performance of HyperMap on different networks
"""
import os
import numpy as np
from collections import defaultdict
import pandas as pd

TARGET_ROOT = os.path.join('../../../data', 'hypermap_results')

HYPERMAP_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'hypermap_{network_name}', 'hypermap_corr_routing_{network_name}_t{temperature}.csv')
HYPERMAP_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'hypermap_corr_routing_all.csv')
HYPER_COORDINATES = os.path.join(TARGET_ROOT, 'coordinates_{network_name}', '{network_name}_{temperature}_hypermap_coordinates.txt')
NETWORK_EDGES = '../../lib/hypercomparison/data/{network_name}_edges.txt'

HYPERMAP_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'hypermap_{network_name}', 'hypermap_link_prediction_{network_name}_t{temperature}.csv')
HYPERMAP_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'hypermap_link_prediction_all.csv')
TEMP_HYPER_COORDINATES = os.path.join(TARGET_ROOT, 'temp_coordinates_{network_name}', '{network_name}_{temperature}_hypermap_coordinates.txt')
TEMP_NETWORK_EDGES = os.path.join(TARGET_ROOT, 'temp_network_edges', '{network_name}_{temperature}_temp_edges.txt')
TEST_EDGES_PATH = os.path.join(TARGET_ROOT, 'temp_network_edges', 'test_edges_{network_name}_{temperature}.csv')
NEGATIVE_EDGES_PATH = os.path.join(TARGET_ROOT, 'temp_network_edges', 'negative_edges_{network_name}_{temperature}.csv')

network_list_all_df = pd.read_csv('../network_list.csv')
#TEMPERATURE = np.round(np.linspace(0.01, 0.99, 25), decimals=2)
NETWORK_NAME = ['karate']
TEMPERATURE = [0.5]

NETWORK_GAMMA_MAPPING = dict(
    network_list_all_df[['network_name', 'gamma']].values
    )

rule run_hypermap_corr_routing_all:
    input:
        expand(HYPERMAP_CORR_ROUTING_RESULT, temperature=TEMPERATURE, network_name=NETWORK_NAME)
    output:
        HYPERMAP_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hypermap_corr_routing:
    input:
        HYPER_COORDINATES
    params:
        '{network_name}', '{temperature}',
        lambda wildcards: str(NETWORK_GAMMA_MAPPING[wildcards.network_name]),
    output:
        HYPERMAP_CORR_ROUTING_RESULT
    shell:
        'python3 demo_hypermap_correlation_greedy_routing.py {input} {params} {output}'

rule run_hyperbolic_embedding:
    input:
        NETWORK_EDGES
    params:
        lambda wildcards: "-g " + str(NETWORK_GAMMA_MAPPING[wildcards.network_name]),
        "-t {temperature}"
    output:
        HYPER_COORDINATES
    shell:
        './hypermap -i {input} {params} -k 10 -o {output} -c no'

#hypermap link prediction
rule run_hypermap_link_prediction_all:
    input:
        expand(HYPERMAP_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, temperature=TEMPERATURE)
    output:
        HYPERMAP_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hypermap_link_prediction:
    input:
        TEMP_HYPER_COORDINATES, TEST_EDGES_PATH, NEGATIVE_EDGES_PATH
    params:
        '{network_name}', '{temperature}',
        lambda wildcards: str(NETWORK_GAMMA_MAPPING[wildcards.network_name])
    output:
        HYPERMAP_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_hypermap_link_prediction.py {input} {params} {output}'

rule run_temp_graph_egdes:
    params:
        '{network_name}', '{temperature}'
    output:
        TEST_EDGES_PATH, NEGATIVE_EDGES_PATH, TEMP_NETWORK_EDGES
    shell:
        'python3 demo_train_test_split.py {params} {output}'

rule run_temp_hyperbolic_embedding:
    input:
        TEMP_NETWORK_EDGES
    params:
        lambda wildcards: "-g " + str(NETWORK_GAMMA_MAPPING[wildcards.network_name]),
        "-t {temperature}"
    output:
        TEMP_HYPER_COORDINATES
    shell:
        './hypermap -i {input} {params} -k 10 -o {output} -c no'