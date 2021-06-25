"""
Snakefile to simulate the performance of hydra on different networks
"""
import os
import pandas as pd

TARGET_ROOT = os.path.join('../../../data', 'hydra')

HYDRA_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'hydra_{network_name}', 'corr_routing_{network_name}_d{dimensions}.csv')
HYDRA_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'hydra_corr_routing_all.csv')

HYDRA_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'hydra_{network_name}', 'new_link_prediction_{network_name}_d{dimensions}.csv')
HYDRA_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'hydra_link_prediction_all.csv')
TEMP_EDGES_PATH = os.path.join(TARGET_ROOT, 'hydra_{network_name}', 'hydra_temp_edges_{network_name}_d{dimensions}.txt')

NETWORK_NAME = ['karate']
#DIMENSIONS = list(range(2, 129, 3)) #dimensions cannot equal 1 for HOPE
DIMENSIONS = [128]

"""
calculate correlation and greedy routing results
"""
rule run_hydra_corr_routing_all:
    input:
        expand(HYDRA_CORR_ROUTING_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        HYDRA_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hydra_corr_routing:
    params:
        '{network_name}', '{dimensions}'
    output:
        HYDRA_CORR_ROUTING_RESULT
    shell:
        'python3 demo_hydra_correlation_greedy_routing.py {params} {output}'

"""
calculate link prediction results
"""
rule run_hydra_link_prediction_all:
    input:
        expand(HYDRA_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        HYDRA_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hydra_link_prediction:
    params:
        '{network_name}', '{dimensions}', TEMP_EDGES_PATH
    output:
        HYDRA_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_hydra_link_prediction.py {params} {output}'
