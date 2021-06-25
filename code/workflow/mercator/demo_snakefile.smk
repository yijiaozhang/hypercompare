"""
Snakefile to simulate the performance of Mercator on different networks
"""
import os
import pandas as pd

TARGET_ROOT = os.path.join('../../../data', 'mercator_results')

MERCATOR_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'mercator_corr_routing', 'corr_routing_{network_name}.csv')
MERCATOR_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'mercator_corr_routing_all.csv')
NETWORK_EDGES = '../../lib/hypercomparison/data/{network_name}_edges.txt'
MERCATOR_COORDINATES = os.path.join(TARGET_ROOT, 'coordinates_mercator', '{network_name}_mercator')
MERCATOR_COORDINATES_DETAIL = os.path.join(TARGET_ROOT, 'coordinates_mercator', '{network_name}_mercator.inf_coord')

MERCATOR_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'mercator_link_prediction', 'link_prediction_{network_name}.csv')
MERCATOR_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'mercator_link_prediction_all.csv')
MERCATOR_LP_COORDINATES = os.path.join(TARGET_ROOT, 'temp_coordinates_mercator', '{network_name}_mercator')
MERCATOR_LP_COORDINATES_DETAIL = os.path.join(TARGET_ROOT, 'temp_coordinates_mercator', '{network_name}_mercator.inf_coord')
TEMP_NETWORK_EDGES = os.path.join(TARGET_ROOT, 'temp_network_edges', '{network_name}_temp_edges.txt')
TEST_EDGES_PATH = os.path.join(TARGET_ROOT, 'temp_network_edges', 'test_edges_{network_name}.csv')
NEGATIVE_EDGES_PATH = os.path.join(TARGET_ROOT, 'temp_network_edges', 'negative_edges_{network_name}.csv')

NETWORK_NAME = ['karate']

rule run_mercator_corr_routing_all:
    input:
        expand(MERCATOR_CORR_ROUTING_RESULT, network_name=NETWORK_NAME)
    output:
        MERCATOR_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_mercator_coordinates:
    input:
        NETWORK_EDGES
    params:
        '{network_name}', MERCATOR_COORDINATES
    output:
        MERCATOR_COORDINATES_DETAIL
    shell:
        'python demo_mercator_coordinates.py {input} {params}'

rule run_mercator_corr_routing:
    input: 
        MERCATOR_COORDINATES_DETAIL
    params:
        '{network_name}'
    output:
        MERCATOR_CORR_ROUTING_RESULT
    shell:
        'python3 demo_mercator_correlation_greedy_routing.py {input} {params} {output}'

# link prediction
rule run_mercator_link_prediction_all:
    input:
        expand(MERCATOR_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME)
    output:
        MERCATOR_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_temp_graph_egdes:
    params:
        '{network_name}', MERCATOR_LP_COORDINATES
    output:
        TEST_EDGES_PATH, NEGATIVE_EDGES_PATH, TEMP_NETWORK_EDGES, MERCATOR_LP_COORDINATES_DETAIL
    shell:
        'python3 demo_mercator_link_prediction_coordinates.py {params} {output}'

rule run_mercator_link_prediction:
    input:
        MERCATOR_LP_COORDINATES_DETAIL, TEST_EDGES_PATH, NEGATIVE_EDGES_PATH
    params:
        '{network_name}'
    output:
        MERCATOR_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_mercator_link_prediction.py {input} {params} {output}'
