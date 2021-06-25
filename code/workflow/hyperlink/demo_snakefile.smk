"""
Snakefile to get the performance of hyperlink on real world networks
"""
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import time

TARGET_ROOT = os.path.join('../../../data', 'hyperlink_results')

HYPERLINK_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'hyperlink_{network_name}', 'hyperlink_corr_routing_{network_name}_t{temperature}.csv')
HYPERLINK_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'hyperlink_corr_routing_all.csv')
HYPER_COORDINATES = os.path.join(TARGET_ROOT, 'coordinates_{network_name}', '{network_name}_{temperature}_hyperlink_coordinates.txt')
NETWORK_EDGES = '../../lib/hypercomparison/data/{network_name}_edges.txt'

HYPERLINK_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'hyperlink_{network_name}', 'hyperlink_link_prediction_{network_name}_t{temperature}.csv')
HYPERLINK_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'hyperlink_link_prediction_all.csv')
TEMP_HYPER_COORDINATES = os.path.join(TARGET_ROOT, 'temp_coordinates_{network_name}', '{network_name}_{temperature}_hyperlink_coordinates.txt')
TEMP_NETWORK_EDGES = os.path.join(TARGET_ROOT, 'temp_network_edges', '{network_name}_{temperature}_temp_edges.txt')
TEST_EDGES_PATH = os.path.join(TARGET_ROOT, 'temp_network_edges', 'test_edges_{network_name}_{temperature}.csv')
NEGATIVE_EDGES_PATH = os.path.join(TARGET_ROOT, 'temp_network_edges', 'negative_edges_{network_name}_{temperature}.csv')

network_list_all_df = pd.read_csv('../network_list.csv')

#TEMPERATURE = np.round(np.linspace(0.1, 0.9, 9), decimals=2)
NETWORK_NAME = ['karate']
TEMPERATURE = [0.3]

NETWORK_GAMMA_MAPPING = dict(
    network_list_all_df[['network_name', 'gamma']].values
    )

rule run_hyperlink_corr_routing_all:
    input:
        expand(HYPERLINK_CORR_ROUTING_RESULT, temperature=TEMPERATURE, network_name=NETWORK_NAME)
    output:
        HYPERLINK_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hyperlink_corr_routing:
    input:
        HYPER_COORDINATES
    params:
        '{network_name}', '{temperature}',
        lambda wildcards: str(NETWORK_GAMMA_MAPPING[wildcards.network_name])
    output:
        HYPERLINK_CORR_ROUTING_RESULT
    shell:
        'python3 demo_hyperlink_correlation_greedy_routing.py {input} {params} {output}'

rule run_hyperlink:
    input:
        NETWORK_EDGES
    params:
        lambda wildcards: str(NETWORK_GAMMA_MAPPING[wildcards.network_name]),
        "{temperature}", 
        str(time.time())
    output:
        HYPER_COORDINATES
    shell:
        './hyperlink.exe {input} {params} {output} 0 20 1'

#hyperlink link prediction
rule run_hyperlink_link_prediction_all:
    input:
        expand(HYPERLINK_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, temperature=TEMPERATURE)
    output:
        HYPERLINK_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hyperlink_link_prediction:
    input:
        TEMP_HYPER_COORDINATES, TEST_EDGES_PATH, NEGATIVE_EDGES_PATH
    params:
        '{network_name}', '{temperature}',
        lambda wildcards: str(NETWORK_GAMMA_MAPPING[wildcards.network_name])
    output:
        HYPERLINK_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_hyperlink_link_prediction.py {input} {params} {output}'

rule run_temp_graph_egdes:
    params:
        '{network_name}', '{temperature}'
    output:
        TEST_EDGES_PATH, NEGATIVE_EDGES_PATH, TEMP_NETWORK_EDGES
    shell:
        'python3 demo_train_test_split.py {params} {output}'

rule run_temp_hyperlink:
    input:
        TEMP_NETWORK_EDGES
    params:
        lambda wildcards: str(NETWORK_GAMMA_MAPPING[wildcards.network_name]),
        "{temperature}", 
        str(time.time())
    output:
        TEMP_HYPER_COORDINATES
    shell:
        './hyperlink.exe {input} {params} {output} 0.3 20 1'
