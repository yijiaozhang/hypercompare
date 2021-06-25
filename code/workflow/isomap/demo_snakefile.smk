"""
Snakefile to simulate the performance of Isomap on different networks
"""
import os
import pandas as pd

TARGET_ROOT = os.path.join('../../../data', 'isomap_results')

ISOMAP_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'isomap_{network_name}', 'corr_routing_{network_name}_d{dimensions}.csv')
ISOMAP_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'isomap_corr_routing_all.csv')

ISOMAP_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'isomap_{network_name}', 'new_link_prediction_{network_name}_d{dimensions}.csv')
ISOMAP_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'isomap_link_prediction_test.csv')
TEMP_EDGES_PATH = os.path.join(TARGET_ROOT, 'isomap_{network_name}', 'isomap_temp_edges_{network_name}_d{dimensions}.txt')

NETWORK_NAME = ['karate']
DIMENSIONS = [128]

rule run_isomap_corr_routing_all:
    input:
        expand(ISOMAP_CORR_ROUTING_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        ISOMAP_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_isomap_corr_routing:
    params:
        '{network_name}', '{dimensions}'
    output:
        ISOMAP_CORR_ROUTING_RESULT
    shell:
        'python3 demo_isomap_correlation_greedy_routing.py {params} {output}'

rule run_isomap_link_prediction_all:
    input:
        expand(ISOMAP_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        ISOMAP_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_isomap_link_prediction:
    params:
        '{network_name}', '{dimensions}', TEMP_EDGES_PATH
    output:
        ISOMAP_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_isomap_link_prediction.py {params} {output}'
