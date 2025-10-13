
from src.explanation_logic.verifier.verifier_base import get_mnist_verifier, get_breast_cancer_verifier

def get_verifier(dataset: str, model_path: str, json_config: str = "configs/bab_config.json",coarsening_timeout_step=12):
    if dataset == "mnist":
        return get_mnist_verifier(model_path=model_path, json_config=json_config,coarsening_timeout_step=coarsening_timeout_step)
    if dataset == "breast_cancer":
        return get_breast_cancer_verifier(timeout=3000,model_path="artifacts/wbc_mlp.onnx",json_config="bab_configs/mnistfc_vnncomp21.json",coarsening_timeout_step=12)


    raise ValueError(f"Unsupported dataset: {dataset}")
