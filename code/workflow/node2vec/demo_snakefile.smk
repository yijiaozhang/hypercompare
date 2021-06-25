"""
Snakefile to simulate the performance of node2vec on different networks
"""
import os
import pandas as pd

TARGET_ROOT = os.path.join('../../../data', 'node2vec_results')

N2V_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'node2vec_{network_name}', 'corr_routing_{network_name}_d{dimensions}_l{walk_length_p}.csv')
N2V_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'node2vec_corr_routing_all.csv')

N2V_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'node2vec_{network_name}', 'new_center_link_prediction_{network_name}_d{dimensions}_l{walk_length_l}.csv')
N2V_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'node2vec_link_prediction_all.csv')

NETWORK_NAME = ['karate']
DIMENSIONS = [128]
WALK_LENGTH_P = [100] #for correlation and greedy routing
WALK_LENGTH_L = [10] #for link prediction

rule run_node2vec_corr_routing_all:
    input:
        expand(N2V_CORR_ROUTING_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS, walk_length_p=WALK_LENGTH_P)
    output:
        N2V_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_node2vec_corr_routing:
    params:
        '{network_name}', '{dimensions}', '{walk_length_p}'
    output:
        N2V_CORR_ROUTING_RESULT
    shell:
        'python3 demo_node2vec_correlation_greedy_routing.py {params} {output}'

rule run_node2vec_link_prediction_all:
    input:
        expand(N2V_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS, walk_length_l=WALK_LENGTH_L)
    output:
        N2V_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_node2vec_link_prediction:
    params:
        '{network_name}', '{dimensions}', '{walk_length_l}'
    output:
        N2V_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_node2vec_link_prediction.py {params} {output}'
