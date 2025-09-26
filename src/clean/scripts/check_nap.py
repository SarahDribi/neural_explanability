

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Iterable, Optional
from clean.verifier.verifier_base import get_mnist_verifier, get_breast_cancer_verifier
from clean.coarsening.nap_coarsen import coarsen_heuristic
from clean.nap_extraction.nap_utils import (
    get_num_kept,
    get_num_total_neurons,
    

)
from clean.nap_extraction.nap_utils import (summarize_nap_single)
from clean.nap_extraction.extract_nap import nap_extraction_from_onnx

from clean.scripts.useful_func import (max_epsilon_robustness,max_epsilon_nap_robustness,get_predicted_label_from_model)
from clean.dataset.utils import get_label_input,get_label_inputs
from clean.dataset.other_dataset import get_correctly_classified_inputs_breast_cancer
from clean.dataset.other_dataset import get_label_inputs_breast_cancer



import csv
import json
import os
import time
#utils
def _count_timeouts(timeout_flags):
    """
    Count number of True flags in possibly nested containers.
    - Handles: None, dict (values), list/tuple/set (possibly nested), scalars.
    - Typical case here: list of lists with booleans at [i][j].
    """
    if timeout_flags is None:
        return 0


    
    if isinstance(timeout_flags, (list, tuple, set)):
        total = 0
        for v in timeout_flags:
            if isinstance(v, (list, tuple, set, dict)):
                total += _count_timeouts(v)
            else:
                total += int(bool(v))
        return total

    
    return int(bool(timeout_flags))


# default params
default_heuristics: List[str] = ["simple","random","preactivation_impact","descending","random1","random2"]
default_model: str = "model.onnx"
SUPPORTED_DATASETS = {"mnist", "breast_cancer"}


# result kept for each heuristic run
@dataclass
class HeuristicResult:
    heuristic_name: str
    input_id: int
    model_name: str
    time_taken: float
    num_neurons_kept: int
    total_neurons: int
    epsilon: float
    epsilon_region: float
    num_timeouts: int
    predicted_label: int
    ground_truth_label: int
    coarsened_nap: Optional[List[List[int]]]  # Nap coarsend using the heuristic traversal
    success: bool  # heuristic terminée dans le délai ?
    other_metrics: Dict[str, Any]  # extensible


@dataclass
class ExplanationInput:
    nap: List[List[int]]
    data: Any
    label: int
    predicted_label: int
    is_correct_class: bool
    epsilon: float
    epsilon_region: float  # optional, -1 if not computed
    model: str
    data_set: str

def summarize_multiple_naps(heuristic_results:list[HeuristicResult], path:str):
    """
    Résume une liste de NAPs (Neural Activation Patterns) associés à différentes heuristiques
    et sauvegarde le tout dans un seul fichier texte.

    Args:
        nap_list (list of tuples): liste de (nap, heuristic_name)
                                   où `nap` est un NAP (list of lists/arrays)
                                   et `heuristic_name` est une chaîne.
        path (str): chemin du fichier texte où sauvegarder le résumé.
    """

    
    all_reports = []
    for result in heuristic_results:
        nap=result.coarsened_nap
        heuristic_name=result.heuristic_name
        if nap is None:
            report = f"=== Neural Activation Pattern Summary ({heuristic_name}) ===\nNo NAP available (heuristic failed or timed out)\n"
            all_reports.append(report)
            all_reports.append("\n" + "="*80 + "\n")
            continue
        report = summarize_nap_single(nap, heuristic_name)
        all_reports.append(report)
        all_reports.append("\n" + "="*80 + "\n")  # séparateur visuel

    final_text = "\n".join(all_reports)

    # Sauvegarde dans un fichier
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"Résumé(s) sauvegardé(s) dans {path}")

def get_explanation_input(
    input_tensor: Any,
    input_index: int,
    model_path: str,
    ground_truth_label: int,
    data_set_name: str,
) -> ExplanationInput:
    # extraction NAP + prédiction
    nap = nap_extraction_from_onnx(model_path, input_tensor)
    #import pdb;pdb.set_trace()
    predicted_label = get_predicted_label_from_model(model_path, input_tensor)

    # vérif dataset + récupérer le vérifieur
    if data_set_name == "breast_cancer":
        if ground_truth_label not in (0, 1):
            raise ValueError(
                f"Ground truth label for breast cancer should be 0 or 1, got {ground_truth_label}"
            )
        verifier = get_breast_cancer_verifier(model_path=model_path,json_config="configs/bab_config.json")
    elif data_set_name == "mnist":
        if ground_truth_label not in range(10):
            raise ValueError(
                f"Ground truth label for mnist should be between 0 and 9, got {ground_truth_label}"
            )
        verifier =get_mnist_verifier(model_path=model_path,json_config="configs/bab_config.json")
    else:
        raise ValueError(f"Unsupported dataset: {data_set_name}")

    # epsilon = robustesse max sous contrainte NAP
    epsilon = max_epsilon_robustness(input_tensor, nap, predicted_label, verifier)
    epsilon_region = max_epsilon_nap_robustness(input_tensor,predicted_label, verifier)
    #
    #print(f"[INFO] Input {input_index} (true label {ground_truth_label}) predicted as {predicted_label} with epsilon {epsilon}")
    #import pdb; pdb.set_trace()

    return ExplanationInput(
        nap=nap,
        data=input_tensor,
        label=ground_truth_label,
        predicted_label=predicted_label,
        is_correct_class=(predicted_label == ground_truth_label),
        epsilon=epsilon,
        epsilon_region=epsilon_region,
        model=model_path,
        data_set=data_set_name,
    )


def real_heuristic_explain_input(
    input_data: ExplanationInput,
    input_id: int,
    data_set_name: str,
    model_name: str,
    heuristic_name: str,
) -> HeuristicResult:
    # choisir le vérifieur
    if data_set_name == "mnist":
        verifier = get_mnist_verifier(model_path=model_name,json_config="configs/bab_config.json")
    elif data_set_name == "breast_cancer":
        verifier = get_breast_cancer_verifier(model_path=model_name,json_config="configs/bab_config.json")
    else:
        raise ValueError(f"Data set not supported: {data_set_name}")

    t0 = time.perf_counter()
    try:
        coarsened_nap, timeout_flags = coarsen_heuristic(
            nap=input_data.nap,
            input=input_data.data,
            label=input_data.predicted_label,
            
            epsilon=input_data.epsilon,
            heuristic_name=heuristic_name,
            verifier=verifier,
            model_path=input_data.model,
        )
        #import pdb; pdb.set_trace()
        other_max_epsilon = max_epsilon_robustness(
            input_data.data, coarsened_nap, input_data.predicted_label, verifier)
        #print(f"Debug: max epsilon after coarsening with  is {other_max_epsilon} before was {input_data.epsilon}")
        #import pdb; pdb.set_trace()
        dt = time.perf_counter() - t0
        

        timeout_count = _count_timeouts(timeout_flags)
        return HeuristicResult(
            heuristic_name=heuristic_name,
            input_id=input_id,
            model_name=model_name,
            time_taken=dt,
            num_neurons_kept=get_num_kept(coarsened_nap),
            total_neurons=get_num_total_neurons(coarsened_nap),
            epsilon=input_data.epsilon,
            epsilon_region=input_data.epsilon_region,
            predicted_label=input_data.predicted_label,
            ground_truth_label=input_data.label,
            num_timeouts=timeout_count,
            coarsened_nap=coarsened_nap,
            success=True,
            other_metrics={"note": "demo"},
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        import pdb; pdb.set_trace()
        print(f"Error in heuristic {heuristic_name} for input {input_id}): {e}")
        return HeuristicResult(
            heuristic_name=heuristic_name,
            input_id=input_id,
            model_name=model_name,
            time_taken=dt,
            num_neurons_kept=0,
            total_neurons=0,
            epsilon=input_data.epsilon,
            epsilon_region= -2.0,
            predicted_label=input_data.predicted_label,
            ground_truth_label=input_data.label,
            num_timeouts=0,
            coarsened_nap=None,
            success=False,
            other_metrics={"error": str(e)},
        )


def extract_info(result: HeuristicResult) -> Dict[str, Any]:
    row = asdict(result)
    other = row.pop("other_metrics", {}) or {}
    for k, v in other.items():
        row[f"m_{k}"] = v
    if result.total_neurons > 0:
        row["kept_ratio"] = result.num_neurons_kept / result.total_neurons
    else:
        row["kept_ratio"] = None
    return row


def run_heuristics_per_input(
    input_data: ExplanationInput,
    input_id: int,
    data_set_name: str,
    model: str = default_model,
    heuristics: Iterable[str] = default_heuristics,
) -> List[HeuristicResult]:
    results: List[HeuristicResult] = []
    for h in heuristics:
        try:
            r = real_heuristic_explain_input(
                input_data=input_data,
                input_id=input_id,
                data_set_name=data_set_name,
                model_name=model,
                heuristic_name=h,
            )
        except Exception as e:
            print(f"Error running heuristic {h} on input {input_id}: {e}")
            r = HeuristicResult(
                heuristic_name=h,
                input_id=input_id,
                model_name=model,
                time_taken=0.0,
                num_neurons_kept=0,
                total_neurons=0,
                epsilon=input_data.epsilon,
                predicted_label=input_data.predicted_label,
                ground_truth_label=input_data.label,
                num_timeouts=1,
                coarsened_nap=None,
                success=False,
                other_metrics={"error": str(e)},
            )
        results.append(r)
    return results


def export_results_as_csv(rows: List[Dict[str, Any]], path: str) -> str:
    if not rows:
        raise ValueError("No rows to export.")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_image_id(label,data_set: str, index: int) -> int:
    if data_set == "mnist":
        return label*1000+index
    elif data_set == "breast_cancer":
        return label*1000+index
    else:
        raise ValueError(f"Unsupported dataset: {data_set}")


def get_full_filename(base_path: str, dataset: str, image_idx: int, filename: str) -> str:
    # éviter 'n°' dans le chemin (compatibilité)
    result_folder = f"{dataset}_image_{image_idx}_explanations"
    path = os.path.join(base_path, result_folder)
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    return full_path


def add_csv_to_results_folder(base_path: str, dataset: str, image_idx: int) -> str:
    return get_full_filename(base_path, dataset, image_idx, "heuristics_results.csv")


def add_image_to_results_folder(base_path: str, dataset: str, image_idx: int) -> str:
    return get_full_filename(base_path, dataset, image_idx, "image.png")


def add_nap_to_results_folder(base_path: str, dataset: str, image_idx: int) -> str:
    return get_full_filename(base_path, dataset, image_idx, "nap.json")


def add_detailed_explanation_to_results_folder(
    base_path: str, dataset: str, image_idx: int, heuristic_name: str
) -> str:
    return get_full_filename(base_path, dataset, image_idx, f"explanation_{heuristic_name}.txt")


def add_analysis_to_results_folder(base_path: str, dataset: str, image_idx: int) -> str:
    return get_full_filename(base_path, dataset, image_idx, "explanation_analysis.txt")


def export_nap_explanations_as_json(results: List[HeuristicResult], path: str):
    rows: List[Dict[str, Any]] = [extract_info(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def export_input(image: Any, path: str):
    # Stub : à implémenter selon ton type d'input.
    # Ex.: si image est un tensor HxW (MNIST), on peut sauvegarder en PNG via PIL/torchvision.
    return


def export_explanation_analysis(results: List[HeuristicResult], path: str):
    analysis = {
        "num_heuristics": len(results),
        "successful_heuristics": sum(1 for r in results if r.success),
        "avg_kept_ratio": (
            sum((r.num_neurons_kept / r.total_neurons) for r in results if r.total_neurons > 0) / max(
                1, sum(1 for r in results if r.total_neurons > 0)
            )
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        for k, v in analysis.items():
            f.write(f"{k}: {v}\n")


def log_results(
    results: List[HeuristicResult], dataset: str, image_idx: int, image: Any, base_path: str
):
    rows: List[Dict[str, Any]] = [extract_info(r) for r in results]
    try:
        csv_path = export_results_as_csv(rows, add_csv_to_results_folder(base_path, dataset, image_idx))
        print(f"CSV écrit: {csv_path}")
    except Exception as e:
        print(f"Export csv failed: {e}")

    try:
        nap_path = add_nap_to_results_folder(base_path, dataset, image_idx)
        export_nap_explanations_as_json(results, nap_path)
        print(f"NAPs écrits: {nap_path}")
    except Exception as e:
        print(f"Export naps failed: {e}")

    try:
        img_path = add_image_to_results_folder(base_path, dataset, image_idx)
        export_input(image, img_path)
        print(f"Image écrite: {img_path}")
    except Exception as e:
        print(f"Export image failed: {e}")

    try:
        analysis_path = add_analysis_to_results_folder(base_path, dataset, image_idx)
        export_explanation_analysis(results, analysis_path)
        print(f"Analyse écrite: {analysis_path}")
    except Exception as e:
        print(f"Export analysis failed: {e}")
    # adding  a detailed explanation for each heuristic
    try:
        detailed_path=get_full_filename(base_path, dataset, image_idx, "detailed_heuristics_explanations.txt")
        summarize_multiple_naps(results,detailed_path)
        print(f"Detailed heuristic explanations written: {detailed_path}")
    except Exception as e:
        print(f"Export detailed heuristic explanations failed: {e}")
    
    print("All result files exported")


def explain_input(
    input_data: ExplanationInput,
    input_id: int,
    data_set_name: str,
    model: str = default_model,
    heuristics: Iterable[str] = default_heuristics,
) -> List[HeuristicResult]:
    results = run_heuristics_per_input(
        input_data=input_data,
        input_id=input_id,
        data_set_name=data_set_name,
        model=model,
        heuristics=heuristics,
    )
    # NB: on loggue aussi l'input brut (image) — ici accessible via input_data.data
    log_results(results, dataset=data_set_name, image_idx=input_id, image=input_data.data, base_path="results")
    return results



def explain_many_images_test_set(num_samples,data_set_name: str = "mnist",onnx_path: str = "models/mnist-10x2.onnx"):
    
    for label in range(10):
        inputs = get_label_inputs(label=label,num_samples=num_samples)
        for img_idx, image in enumerate(inputs):
            ground_truth_label = label
            image_id = generate_image_id(label,data_set_name, img_idx)

            input_explanation = get_explanation_input(
                input_tensor=image,
                input_index=img_idx,
                model_path=onnx_path,
                ground_truth_label=ground_truth_label,
                data_set_name=data_set_name,
            )
            if input_explanation.epsilon_region==input_explanation.epsilon:
                print(f"Skipping image {image_id} as epsilon_region equals epsilon")
                continue
            heuristics = default_heuristics
            _ = explain_input(
                input_data=input_explanation,
                input_id=image_id,
                data_set_name=data_set_name,
                model=onnx_path,
                heuristics=heuristics,
            )
    print("All images explained")
def explain_breast_cancer_images(num_to_explain: int = 1,data_set_name: str = "breast_cancer",onnx_path: str = "artifacts/wbc_mlp.onnx"):

    for label in [0,1]:
        inputs = get_label_inputs_breast_cancer(label=label,num_samples=num_to_explain)
        for img_idx, image in enumerate(inputs):
            ground_truth_label = label
            image_id = generate_image_id(label,data_set_name, img_idx)

            input_explanation = get_explanation_input(
                input_tensor=image,
                input_index=img_idx,
                model_path=onnx_path,
                ground_truth_label=ground_truth_label,
                data_set_name=data_set_name,
            )
            if input_explanation.epsilon_region==input_explanation.epsilon:
                print(f"Skipping image {image_id} as epsilon_region equals epsilon")
                continue
            heuristics = default_heuristics
            _ = explain_input(
                input_data=input_explanation,
                input_id=image_id,
                data_set_name=data_set_name,
                model=onnx_path,
                heuristics=heuristics,
            )
    print("All breast cancer images explained")

import argparse
def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Explain images using heuristics.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        default="mnist",
        help="Dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/mnist-10x2.onnx",
        help="Path to the ONNX model (default: models/mnist-10x2.onnx)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to explain per class (default: 1)",
    )
    args = parser.parse_args()
    return args

