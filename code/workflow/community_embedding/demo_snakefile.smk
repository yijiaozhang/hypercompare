"""
Snakefile to simulate the performance of community embedding on different networks
"""
import os
import numpy as np

TARGET_ROOT = os.path.join('../../../data', 'community_results')

COMMUNITY_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'community_{network_name}', 'corr_routing_{network_name}_beta{beta}.csv')
COMMUNITY_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'community_corr_routing_all.csv')

COMMUNITY_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'community_{network_name}', 'link_prediction_{network_name}_beta{beta}.csv')
COMMUNITY_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'community_link_prediction_all.csv')
COMMUNITY_TEMP_EDGES_PATH = os.path.join(TARGET_ROOT, 'community_{network_name}', 'community_temp_edges_{network_name}_beta{beta}.txt')

NETWORK_NAME = ['karate']
#BETA = np.round(np.linspace(0.1, 0.9, 9), decimals=2)
BETA = [0.3]

"""
calculate correlation and greedy routing results on different networks using community embedding 
"""
rule run_community_corr_routing_all:
    input:
        expand(COMMUNITY_CORR_ROUTING_RESULT, network_name=NETWORK_NAME, beta=BETA)
    output:
        COMMUNITY_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_community_corr_routing:
    params:
        '{network_name}', '{beta}'
    output:
        COMMUNITY_CORR_ROUTING_RESULT
    shell:
        'python3 demo_community_correlation_greedy_routing.py {params} {output}'

"""
calculate link prediction results on different networks using community embedding
"""
rule run_community_link_prediction_all:
    input:
        expand(COMMUNITY_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, beta=BETA)
    output:
        COMMUNITY_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_community_link_prediction:
    params:
        '{network_name}', '{beta}', COMMUNITY_TEMP_EDGES_PATH
    output:
        COMMUNITY_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_community_link_prediction.py {params} {output}'
