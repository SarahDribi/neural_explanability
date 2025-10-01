




from src.explanation_logic.verification_utils.inner_bounds_compute import compute_bounds_with_nap_around_input




import argparse
import torch
import json
import copy
from plnn.proxlp_solver.propagation import Propagation 
from src.tools.bab_tools import vnnlib_utils
from src.tools.bab_tools.bab_runner import bab_from_json, bab_output_from_return_dict
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
import plnn.branch_and_bound.utils as bab_utils

import time
from src.explanation_logic.verification_utils.inner_bounds_compute import compute_bounds_around_input



def verify_robustness_around_input(model_path,input,epsilon, json_config, label, use_gpu,timeout,num_classes=10):
    layers, domain,lbs,ubs = compute_bounds_around_input(model_path,input,epsilon, label, use_gpu,num_classes=num_classes)
    bab=False


    if bab:
        if json_config:
            with open(json_config, "r") as f:
                config = json.load(f)
        else:
            print("[INFO] Using default BaB config")
            config = {
        "batch_size": 2000,
        "initial_max_domains": 500,
        "decision_thresh": 0.1,
        "score_function": "kfsb",
        "sort_domain_interval": 5,
        "max_domains": 10000,
        "branching": "relu_heuristic",
        "bound_prop_method": {
            "root": {
                "best": {}
            }
        },
        "cut": False
    }


        return_dict = {}
        print("\n[INFO] Running BaB...")
        bab_from_json(config, layers, domain, return_dict,
                    nn_name="mnist-nap", instance_timeout=500, gpu=use_gpu,precomputed_ibs=(lbs,ubs))
        del config  # prevent reuse issues

        result, nodes = bab_output_from_return_dict(return_dict)
        print(f"\n[RESULT] BaB verification status: {result}")
        print(f"[INFO] BaB visited {nodes} nodes")
        # I need to run the verification with bab first and then run with the milp

    else:
        n_threads=3 # 3 threads par defaut
        anderson_mip_net = AndersonLinearizedNetwork(
                layers, mode="mip-exact", decision_boundary=0.0)
        #  add a flag of usingNap to TRUE or False as a class attribute
        #  add this FLAG to the vlass
        """
        add a propaga
       
        """
        
        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab_utils.subproblems_to_cpu(
            domain.unsqueeze(0), lbs, ubs, squeeze=True)
        
        print(f"The cpu_domain is {cpu_domain}")
        print(cpu_domain.shape)
        
        anderson_mip_net.build_model_using_bounds(cpu_domain[0], (cpu_intermediate_lbs, cpu_intermediate_ubs),
                                                    n_threads=3)
        
        sat_status, global_lb, bab_nb_states = anderson_mip_net.solve_mip(timeout=timeout, insert_cuts=False)
        
        #import pdb; pdb.set_trace()
        return not sat_status


    



def branch_and_bound_verif(model_path,input,epsilon, json_config, label, use_gpu,timeout,num_classes=10):
    layers, domain,lbs,ubs = compute_bounds_around_input(model_path,input,epsilon, label, use_gpu,num_classes=num_classes)
    bab=True


    if bab:
        if json_config:
            with open(json_config, "r") as f:
                config = json.load(f)
        else:
            print("[INFO] Using default BaB config")
            config = {
        "batch_size": 2000,
        "initial_max_domains": 500,
        "decision_thresh": 0.1,
        "score_function": "kfsb",
        "sort_domain_interval": 5,
        "max_domains": 10000,
        "branching": "relu_heuristic",
        "bound_prop_method": {
            "root": {
                "best": {}
            }
        },
        "cut": False
    }


        return_dict = {}
        print("\n[INFO] Running BaB...")
        bab_from_json(config, layers, domain, return_dict,
                    nn_name="mnist-nap", instance_timeout=500, gpu=use_gpu,precomputed_ibs=(lbs,ubs))
        del config  # prevent reuse issues

        result, nodes = bab_output_from_return_dict(return_dict)
        
        
        print(f"[INFO] BaB visited {nodes} nodes")
        verification_status=True
        # If we find a counter example or it says unknown se say it isnt robust
        if result=='True' or result=='ET':
            verification_status=False
            print(f"\n[RESULT] BaB verification status: {verification_status}")


        return  verification_status
  