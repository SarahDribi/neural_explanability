
from clean.nap_extraction.extract_nap import nap_extraction_from_onnx
#from clean.nap_extraction.extract_nap_changed import nap_extraction_from_onnx_breast_cancer
from clean.dataset.utils import get_label_inputs
from clean.dataset.other_dataset import get_correctly_classified_inputs_breast_cancer
from clean.nap_extraction.extract_nap import load_model
import sys

def max_epsilon_robustness(x, nap, label, verifier, low=0.001, high=1, tol=0.001, max_iter=100):
    if verifier.is_verified_nap(nap,x, label,high):
        return high
    for _ in range(max_iter):
        
        mid_epsilon=(low + high) / 2
        if verifier.is_verified_nap(nap,x, label, mid_epsilon):
            low = mid_epsilon
        else:
            high = mid_epsilon
        if high - low < tol:
            break
    if not verifier.is_verified_nap(nap,x, label, low):
        print("[WARN Nap ]Sorry to tell that it was not found to be robust even here {low}")
        return 0
    return low


"""




"""
def max_epsilon_nap_robustness(x, label, verifier, low=0.001, high=1, tol=0.001, max_iter=10):
    if verifier.is_verified_region(x, label, high):
        return high
    for _ in range(max_iter):
        
        mid_epsilon=(low + high) / 2
        if verifier.is_verified_region(x, label, mid_epsilon):
            low = mid_epsilon
        else:
            high = mid_epsilon
        if high - low < tol:
            break
    if not verifier.is_verified_region(x, label, low):
        print(f"[WARN]Sorry to tell that it was not found to be robust even here {low}")
        return 0
    return low
 

def max_epsilon_bab(x, label, verifier, low=0.001, high=1, tol=0.001, max_iter=3):
    for _ in range(max_iter):
        
        mid_epsilon=(low + high) / 2
        if verifier.verif_without_milp(x, label, mid_epsilon):
            low = mid_epsilon
        else:
            high = mid_epsilon
        if high - low < tol:
            break
    if not verifier.verif_without_milp(x, label, low):
        print(f"[WARN]Sorry to tell that it was not found to be robust even here {low}")
        return 0
    return low
 

    










 # finding good inputs that have certain robustness properties 


def get_nap_specification_exclusive(
    label: int,
    verifier,
    model,
    args,
    device: str = "cpu",
    delta: float = 0.001,
    return_all: bool = False
):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    inputs = get_label_inputs(label)  
    print(f"[INFO] Searching for NAP-exclusive robust inputs for label {label}")

    results = []

    for i in range(len(inputs)):
        image_input = inputs[i].to(device).unsqueeze(0)

        
        prediction = model(image_input).argmax(dim=1).item()
        if prediction != label:
            continue

        
        nap = nap_extraction_from_onnx(args.model, image_input.squeeze(0))

        
        max_eps_nap = max_epsilon_robustness(
            image_input, nap, label, verifier,
            low=0.001, high=1.0, tol=0.0001, max_iter=300
        )

        
        max_eps_base = max_epsilon_nap_robustness(
            image_input, label, verifier,
            low=0.001, high=1.0, tol=0.0001, max_iter=300
        )


        if max_eps_nap > max_eps_base + delta:
            print(f" Found input where NAP improves robustness! at index {i} epsilons are  {max_eps_nap} and  {max_eps_base}")
            result = (image_input.squeeze(0).cpu(), max_eps_nap,nap )
            if not return_all:
                return result
            else:
                results.append(result)

    if return_all:
        return results
    else:
        print("No example found where NAP strictly improves robustness.")
        return None

   

def get_nap_specification_exclusive_breast_cancer(
    label: int,
    verifier,
    model,
    args,
    device: str = "cpu",
    delta: float = 0.001,
    return_all: bool = False
):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    inputs =get_correctly_classified_inputs_breast_cancer(label)
    print(f"[INFO] Searching for NAP-exclusive robust inputs for label {label}")

    results = []

    for i in range(len(inputs)):
        image_input=inputs[i]
        nap = nap_extraction_from_onnx(args.model, image_input.squeeze(0))

        
        max_eps_nap = max_epsilon_robustness(
            image_input, nap, label, verifier,
            low=0.001, high=1.0, tol=0.0001, max_iter=300
        )

        
        max_eps_base = max_epsilon_nap_robustness(
            image_input, label, verifier,
            low=0.001, high=1.0, tol=0.0001, max_iter=300
        )


        if max_eps_nap > max_eps_base + delta:
            print(f" Found input where NAP improves robustness! at index {i} epsilons are  {max_eps_nap} and  {max_eps_base}")
            result = (image_input.squeeze(0).cpu(), max_eps_nap,nap )
            if not return_all:
                return result
            else:
                results.append(result)

    if return_all:
        return results
    else:
        print("No example found where NAP strictly improves robustness.")
        return None

   

def get_predicted_label_from_model(model_path,input_tensor):
   
    # Load model
    try:
        model= load_model(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        model.eval()
        predicted_label = model(input_tensor.unsqueeze(0)).argmax(dim=1).item()

    except Exception as e:
        print(f"[ERROR] Failed to get predicted info: {e}", file=sys.stderr)
        sys.exit(1)
    return int(predicted_label)