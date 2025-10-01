import argparse
import torch
import json
import copy
from plnn.proxlp_solver.propagation import Propagation
from src.tools.bab_tools.model_utils import one_vs_all_from_model
from src.tools.bab_tools import vnnlib_utils
import time
from src.explanation_logic.verification_utils.nap_inner_bounds_enforcing import update_bounds_with_nap
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "oval-bab", "tools", "bab_tools"))
from src.tools.bab_tools import vnnlib_utils
from src.tools.bab_tools.model_utils import add_single_prop


    


def other_label(label):
    return (label+1)%2



def compute_bounds_with_nap_around_input(model_path,input,epsilon, label, nap, use_gpu,num_classes=10):
    print(f"[INFO] Loading ONNX model from {model_path}")
    model, in_shape, out_shape, dtype, model_correct = vnnlib_utils.onnx_to_pytorch(model_path)
    assert model_correct, "ONNX model conversion mismatch."
    assert vnnlib_utils.is_supported_model(model), "Model structure unsupported."

    
    center_img = input
    input_point = center_img.to(torch.float32)
   
    input_bounds = torch.stack([(input_point - epsilon).clamp(0, 1), (input_point + epsilon).clamp(0, 1)], dim=-1)
    if epsilon==-1: #whole input space
         input_bounds = torch.stack([torch.zeros_like(input_point), torch.ones_like(input_point)], dim=-1)


    layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype)
    if num_classes>2:
        net = one_vs_all_from_model(
            torch.nn.Sequential(*layers),
            label,
            domain=input_bounds,
            use_ib=True,
            
            gpu=use_gpu,
            num_classes=num_classes
        )
    else:
        net=add_single_prop(layers,label,other_label(label),num_classes=2)
    domain_batch = input_bounds.unsqueeze(0)
    if use_gpu:
        net = [layer.cuda() for layer in net]
        domain_batch = domain_batch.cuda()

    prop = Propagation(net, type="best_prop", params={"best_among": ["KW", "crown"]})
    with torch.no_grad():
        prop.define_linear_approximation(domain_batch)

    lbs = prop.lower_bounds
    ubs = prop.upper_bounds

    print("\n[DEBUG] Layer-wise bound shapes BEFORE NAP:")
    for i, (lb, ub) in enumerate(zip(lbs, ubs)):
        print(f"  Layer {i}: LB shape = {lb.shape}, UB shape = {ub.shape}")

    lbs_orig = copy.deepcopy(lbs)
    ubs_orig = copy.deepcopy(ubs)

    
    print("\n[DEBUG] Applying NAP to adjust inner bounds print ubs lbs , nap...")
    update_bounds_with_nap(ubs, lbs, nap)
    # 
    # I need to add a boolen table to indicate wich neurons activation status has been changed by nap
    # then I can use this table to avoid nap changes in the next iterations

    print("[DEBUG] NAP adjustment complete.")
    

    print("\n[DEBUG] Layer-wise bounds AFTER applying NAP:")
    for i, (lb, ub) in enumerate(zip(lbs, ubs)):
        print(f"  Layer {i}: LB min={lb.min():.4f}, max={lb.max():.4f} | UB min={ub.min():.4f}, max={ub.max():.4f}")

    

    
    return net, domain_batch,lbs,ubs

   






def compute_bounds_around_input(model_path,input,epsilon, label, use_gpu,num_classes=10):
    print(f"[INFO] Loading ONNX model from {model_path}")
    model, in_shape, out_shape, dtype, model_correct = vnnlib_utils.onnx_to_pytorch(model_path)
    assert model_correct, "ONNX model conversion mismatch."
    assert vnnlib_utils.is_supported_model(model), "Model structure unsupported."

        # Load one example image from the dataset with the correct label
    center_img = input
    input_point = center_img.to(torch.float32)
   
    input_bounds = torch.stack([(input_point - epsilon).clamp(0, 1), (input_point + epsilon).clamp(0, 1)], dim=-1)
    if epsilon==-1: #whole input space
         input_bounds = torch.stack([torch.zeros_like(input_point), torch.ones_like(input_point)], dim=-1)


    layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype)

    if num_classes>2:
        net = one_vs_all_from_model(
            torch.nn.Sequential(*layers),
            label,
            domain=input_bounds,
            use_ib=True,
            
            gpu=use_gpu,
            num_classes=num_classes
        )
    else:
        net=add_single_prop( layers,label,other_label(label),num_classes=2)

    domain_batch = input_bounds.unsqueeze(0)
    if use_gpu:
        net = [layer.cuda() for layer in net]
        domain_batch = domain_batch.cuda()

    prop = Propagation(net, type="best_prop", params={"best_among": ["KW", "crown"]})
    with torch.no_grad():
        prop.define_linear_approximation(domain_batch)

    lbs = prop.lower_bounds
    ubs = prop.upper_bounds

    
    

    return net,domain_batch,lbs,ubs
