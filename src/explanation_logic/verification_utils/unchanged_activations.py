
from typing import List
import torch
from src.explanation_logic.verification_utils.inner_bounds_compute import compute_bounds_around_input
from src.explanation_logic.nap_extraction.extract_nap import nap_extraction_from_onnx
# I am deleting those afterwards 
from src.explanation_logic.dataset.utils import get_label_input
# just like update bounds with nap
def find_fixed_neuron_states(
    ubs: List[torch.Tensor], 
    lbs: List[torch.Tensor],
    nap: List[List[int]]
) -> List[List[int]]:

    fixed_states = []
    num_layers_latent=len(nap)
    # we start from one because 1 because 
    for i in range(1, num_layers_latent+1):
        layer_neuron_lower_bounds = lbs[i][0]
        layer_neuron_upper_bounds = ubs[i][0]
        
        
        is_fixed = (layer_neuron_lower_bounds * layer_neuron_upper_bounds > 0).int().tolist()
        fixed_states.append(is_fixed)
        
    return fixed_states

def write_as_str(activations:List[List[int]]):
    """
    activations: list of lists
        Each inner list represents a layer.
        1 = neuron fixed, 0 = neuron not fixed.
    """
    report = []
    for layer_idx, layer in enumerate(activations):
        for neuron_idx, status in enumerate(layer):
            if status == 1:
                report.append(f"Layer {layer_idx}, Neuron {neuron_idx}: fixed")
            elif status == 0:
                report.append(f"Layer {layer_idx}, Neuron {neuron_idx}: not fixed")
            else:
                report.append(f"Layer {layer_idx}, Neuron {neuron_idx}: invalid status {status}")
    return "\n".join(report)



def export_unchanged_activations(activations: List[List[int]], file_path: str = None):
    report = write_as_str(activations)

    if file_path:  
        with open(file_path, "w") as f:
            f.write(report)

    return 


def get_fixed_state_neurons_count(res):
    count=0
    for layer in res:
        count += sum(layer)

    return count

# faudra juste avoir les lower_bounds ,upper_bounds

def get_unchanged_activations(model_path,input_tensor,epsilon,nap,label,num_classes=10):
     net,domain_batch,lbs,ubs=compute_bounds_around_input(model_path,input_tensor,epsilon,label,use_gpu=False,num_classes=num_classes)
     fixed_state_neurons=find_fixed_neuron_states(ubs,lbs,nap)
     return fixed_state_neurons


# I am now going to try it with main
# And as a sanity test they should augment when epsilon narrows

# I should test that if EPSILON IS SMALL all of them get fixed
# else if epsilon=-1 => whole input space none of them is

