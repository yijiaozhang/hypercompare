 """
Snakefile to simulate the performance of poincare maps on different networks
"""
import os
import numpy as np
import pandas as pd

TARGET_ROOT = os.path.join('../../../data', 'poincare_maps_results')

POINMAP_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'poincare_maps_{network_name}', 'poinmaps_corr_routing_{network_name}.csv')
POINMAP_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'poincare_maps_corr_routing_all.csv')
POINMAP_COORDINATES = os.path.join(TARGET_ROOT, 'poincare_maps_{network_name}', '{network_name}_poinmaps_coordinates')

POINMAP_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'poincare_maps_{network_name}', 'poinmaps_link_prediction_{network_name}.csv')
POINMAP_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'poincare_maps_link_prediction_all.csv')
TEMP_POINMAP_COORDINATES = os.path.join(TARGET_ROOT, 'poincare_maps_{network_name}', 'temp_{network_name}_poinmaps_coordinates')
TEMP_EDGES = os.path.join(TARGET_ROOT, 'poincare_maps_{network_name}', 'temp_{network_name}_poinmaps_edges')
TEST_EDGES = os.path.join(TARGET_ROOT, 'poincare_maps_{network_name}', 'temp_{network_name}_poinmaps_test')
NEGATIVE_EDGES = os.path.join(TARGET_ROOT, 'poincare_maps_{network_name}', 'temp_{network_name}_poinmaps_negative')

NETWORK_NAME = ['karate']

rule run_poinmap_corr_routing_all:
    input:
        expand(POINMAP_CORR_ROUTING_RESULT, network_name=NETWORK_NAME)
    output:
        POINMAP_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_poinmap_corr_routing:
    params:
        '{network_name}', POINMAP_COORDINATES
    output:
        POINMAP_CORR_ROUTING_RESULT
    shell:
        'python3 demo_poinmaps_correlation_greedy_routing.py {params} {output}'

rule run_poinmap_link_prediction_all:
    input:
        expand(POINMAP_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME)
    output:
        POINMAP_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_poinmap_link_prediction:
    params:
        '{network_name}', TEMP_POINMAP_COORDINATES, TEMP_EDGES, TEST_EDGES, NEGATIVE_EDGES
    output:
        POINMAP_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_poinmaps_link_prediction.py {params} {output}'