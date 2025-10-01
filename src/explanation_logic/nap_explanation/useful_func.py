
from src.explanation_logic.nap_extraction.extract_nap import nap_extraction_from_onnx
#from src.explanation_logic.nap_extraction.extract_nap_changed import nap_extraction_from_onnx_breast_cancer

from src.explanation_logic.nap_extraction.extract_nap import load_model
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