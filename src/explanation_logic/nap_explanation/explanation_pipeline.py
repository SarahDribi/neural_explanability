# src/napx/pipeline.py
import logging
from pathlib import Path
from typing import Iterable, List
# I d better use the heuristics from the heuristics folder
from src.explanation_logic.nap_explanation.explanation_types import DEFAULT_HEURISTICS, ExplanationInput, HeuristicResult
from src.explanation_logic.utils.io_utils import make_run_dir, class_dir, export_csv, export_json, export_tensor
from src.explanation_logic.nap_extraction.extract_nap import nap_extraction_from_onnx
from src.explanation_logic.nap_extraction.nap_utils import summarize_nap_single
from src.explanation_logic.verification_utils.unchanged_activations import get_unchanged_activations, export_unchanged_activations
from src.explanation_logic.nap_explanation.useful_func import max_epsilon_robustness, max_epsilon_nap_robustness, get_predicted_label_from_model
from src.explanation_logic.nap_explanation.verifier import get_verifier
from src.explanation_logic.nap_explanation.heuristics_runner import run_many_heuristics
log = logging.getLogger(__name__)

def build_explanation_input(input_tensor, gt_label: int, dataset: str, model_path: str) -> ExplanationInput:
    nap = nap_extraction_from_onnx(model_path, input_tensor)
    pred = get_predicted_label_from_model(model_path, input_tensor)
    if dataset == "breast_cancer" and gt_label not in (0, 1):
        raise ValueError("Breast-cancer label must be 0/1")
    if dataset == "mnist" and not (0 <= gt_label <= 9):
        raise ValueError("MNIST label must be in [0..9]")

    verifier = get_verifier(dataset, model_path)
    eps = max_epsilon_robustness(input_tensor, nap, pred, verifier)
    eps_region = max_epsilon_nap_robustness(input_tensor, pred, verifier)
    try:
        num_classes = 2 if dataset == "breast_cancer" else 10
        region = get_unchanged_activations(model_path, input_tensor, eps, nap, pred, num_classes=num_classes)
    except Exception as e:
        log.warning("Failed unchanged activations: %s", e)
        region = []

    return ExplanationInput(
        nap=nap, data=input_tensor, label=gt_label, predicted_label=pred,
        is_correct_class=(pred == gt_label), epsilon=eps, epsilon_region=eps_region,
        model=model_path, data_set=dataset, region_constrained_activations=region
    )

def export_all(run_dir: Path, dataset: str, image_idx: int,
               results: List[HeuristicResult], exp: ExplanationInput):
    d = class_dir(run_dir, dataset, image_idx)
    export_csv([r.to_row() for r in results], d / "heuristics_results.csv")
    export_json([r.to_row() for r in results], d / "nap.json")
    export_tensor(exp.data, d / "input.pt")

    # summary + analysis
    try:
        texts = []
        for r in results:
            if r.coarsened_nap is None: 
                texts.append(f"=== {r.heuristic_name} ===\nNo NAP available\n")
            else:
                texts.append(summarize_nap_single(r.coarsened_nap, r.heuristic_name) + "\n")
        (d / "detailed_heuristics_explanations.txt").write_text("\n" + "="*80 + "\n".join(texts), encoding="utf-8")
    except Exception as e:
        log.warning("Failed to write detailed explanations: %s", e)

    try:
        export_unchanged_activations(exp.region_constrained_activations, d / "constrained_activations_by_region.txt")
    except Exception as e:
        log.warning("Failed to export unchanged activations: %s", e)



# I also need to propagate the coarsening_timeout_step
def explain_input(
    exp: ExplanationInput, input_id: int, dataset: str, model: str,
    heuristics: Iterable[str] = DEFAULT_HEURISTICS, outputs_base: str = "outputs", tag: str = ""
,coarsening_timeout_step=12) -> List[HeuristicResult]:
 
    run_dir = make_run_dir(outputs_base, tag or f"{dataset}_img{input_id}")
    results = run_many_heuristics(exp, input_id, dataset, model, heuristics,coarsening_timeout_step=coarsening_timeout_step)
    export_all(run_dir, dataset, input_id, results, exp)
    return results
