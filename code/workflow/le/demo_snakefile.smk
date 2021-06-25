"""
Snakefile to simulate the performance of Laplacian Eigenmap (LE) on different networks
"""
import os
import pandas as pd

TARGET_ROOT = os.path.join('../../../data', 'le_results')

LE_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'le_{network_name}', 'corr_routing_{network_name}_d{dimensions}.csv')
LE_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'le_corr_routing_all.csv')

LE_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'le_{network_name}', 'new_link_prediction_{network_name}_d{dimensions}.csv')
LE_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'le_link_prediction_test.csv')

NETWORK_NAME = ['karate']
DIMENSIONS = [128]

rule run_le_corr_routing_all:
    input:
        expand(LE_CORR_ROUTING_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        LE_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_le_corr_routing:
    params:
        '{network_name}', '{dimensions}'
    output:
        LE_CORR_ROUTING_RESULT
    shell:
        'python3 demo_le_correlation_greedy_routing.py {params} {output}'

rule run_le_link_prediction_all:
    input:
        expand(LE_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        LE_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_le_link_prediction:
    params:
        '{network_name}', '{dimensions}'
    output:
        LE_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_le_link_prediction.py {params} {output}'
