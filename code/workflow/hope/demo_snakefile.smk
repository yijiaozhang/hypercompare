"""
Snakefile to simulate the performance of HOPE on different networks
"""
import os

TARGET_ROOT = os.path.join('../../../data', 'hope_results')

HOPE_CORR_ROUTING_RESULT = os.path.join(TARGET_ROOT, 'hope_{network_name}', 'corr_routing_{network_name}_d{dimensions}.csv')
HOPE_CORR_ROUTING_RESULT_ALL = os.path.join(TARGET_ROOT, 'hope_corr_routing_all.csv')

HOPE_LINK_PREDICTION_RESULT = os.path.join(TARGET_ROOT, 'hope_{network_name}', 'link_prediction_{network_name}_d{dimensions}.csv')
HOPE_LINK_PREDICTION_RESULT_ALL = os.path.join(TARGET_ROOT, 'hope_link_prediction_all.csv')

NETWORK_NAME = ['karate']
#DIMENSIONS = list(range(2, 129, 3)) #dimensions cannot equal 1 for HOPE
DIMENSIONS = [128]

"""
calculate correlation and greedy routing results
"""
rule run_hope_corr_routing_all:
    input:
        expand(HOPE_CORR_ROUTING_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        HOPE_CORR_ROUTING_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hope_corr_routing:
    params:
        '{network_name}', '{dimensions}'
    output:
        HOPE_CORR_ROUTING_RESULT
    shell:
        'python3 demo_hope_correlation_greedy_routing.py {params} {output}'

"""
calculate link prediction results
"""
rule run_hope_link_prediction_all:
    input:
        expand(HOPE_LINK_PREDICTION_RESULT, network_name=NETWORK_NAME, dimensions=DIMENSIONS)
    output:
        HOPE_LINK_PREDICTION_RESULT_ALL
    shell:
        'python ../merge_results.py {input} {output}'

rule run_hope_link_prediction:
    params:
        '{network_name}', '{dimensions}'
    output:
        HOPE_LINK_PREDICTION_RESULT
    shell:
        'python3 demo_hope_link_prediction.py {params} {output}'
