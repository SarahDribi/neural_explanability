

import torch
import copy
from tqdm import tqdm
from typing import List
    




def update_bounds_with_nap(ubs, lbs, nap):
    print(f"[DEBUG] Total bound layers: {len(ubs)} (includes input and output)")
    for i in range(1, len(ubs) - 1):  # skip first layer to start applying Nap update
        nap_layer_index = i - 1
        if nap_layer_index >= len(nap): # if the nap array len is reached 
            #it means that we are on the  output / extra verification layers
            print(f"NAP update only for inner bounds, skipping because this is either an  output or/an extra verif layer.")
            continue

        layer_nap = nap[nap_layer_index]
        num_neurons = ubs[i].shape[-1]
        print(f"[DEBUG] Intermidiate Layer {i}: NAP length = {len(layer_nap)}, Bounds neurons = {num_neurons}")

        for j in range(min(len(layer_nap), num_neurons)):
            print(f"[DEBUG] Now working on the {j} th neuron of {i} th relu layer ")
            if layer_nap[j] == 1:
                print(f"    [ACTive neuron in Nap] Layer {i}, Neuron {j}")
                lbs[i][0][j] = torch.max(torch.tensor(0.0, device=lbs[i].device), lbs[i][0][j] )
            elif layer_nap[j] == 0:
                print(f"    [Inactive neuron in Nap] Layer {i}, Neuron {j}")
                ubs[i][0][j] = torch.min( ubs[i][0][j],torch.tensor(0.0,device=ubs[i].device))
                   
            else:
                print(f"    [DEBUG] Neuron {j} in Layer {i} not affected by NAP (nap value: {layer_nap[j]})")



"""
    Analyzes neuron bounds to identify neurons with a fixed activation state.

    A neuron's state is fixed if its pre-activation bounds are either both non-negative
    (always active) or both non-positive (always inactive).

    Args:
        ubs: A list of upper-bound tensors for each layer.
        lbs: A list of lower-bound tensors for each layer.
        nap : to be able to dectect only the latent layers

    Returns:
        A list of lists, where each inner list corresponds to a hidden layer.
        An element is 1 if the neuron's state is fixed, and 0 otherwise.
"""
