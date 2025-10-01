from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Iterable
from src.explanation_logic.verifier.verifier_base import get_mnist_verifier, get_breast_cancer_verifier
from src.explanation_logic.coarsening.nap_coarsen import coarsen_heuristic
from src.explanation_logic.nap_extraction.nap_utils import get_num_kept,get_num_total_neurons,get_predicted_label_from_model
from src.explanation_logic.nap_extraction.extract_nap import nap_extraction_from_onnx
from src.explanation_logic.scripts.useful_func import max_epsilon_robustness
from src.explanation_logic.dataset.utils import get_label_input

import csv
import time

# default params
default_heuristics: List[str] = ["simple"]
default_model: str = "model.onnx"
SUPPORTED_DATASETS = {"mnist", "breast_cancer"}


"""
To make a dataset supported What we  need is :
- define how to load it (images, labels)
- define if there is a preprocessing step (eg. normalization)
- Have an  (ONNX) model that is compatible with the dataset
and supported by the verifier : Oval bab code 

"""


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
    num_timeouts: int
    predicted_label: int
    ground_truth_label:int 
    success: bool  # heuristic terminée dans le délai ?
    other_metrics: Dict[str, Any]  # extensible

"""
A little description of the input data needed to explain an input

"""
"""
The explanation of the model's decision on a given input is local,
in our case we want to know the most determinant neurons conducting to the decision 
in that local region to that input and depends on:
- The input data itself (e.g., an image or a feature vector).
- The ground truth label of the input (if available).
- the model itself is passed because we need to extract the neural activation pattern (NAP)
- The model's predicted label for that input.
- the epsilon value used for robustness verification depending on the dataset and the model
- 

"""
@dataclass
class ExplanationInput:
    nap: List[List[int]]
    data: Any
    label: int
    predicted_label: int
    is_correct_class: bool
    epsilon: float
    model: str
    data_set: str   

# Function that explains an input model's decision using a heuristic and returns a HeuristicResult.
#
def get_explanation_input(input_tensor,input_index,model_path,ground_truth_label:int,data_set_name:str) -> Dict[str, Any]:

    nap=nap_extraction_from_onnx(model_path,input_tensor)
    predicted_label = get_predicted_label_from_model(model_path, input_tensor)
    # epsilon is considered as a value between 0 and 1
    # here I set it as the maximum perturbation where the model is still robust
    # given the nap constraint
    if data_set_name=="breast_cancer":
        verifier=get_breast_cancer_verifier()
        if ground_truth_label not in [0,1]:
            raise ValueError(f"Ground truth label for breast cancer should be 0 or 1, got {ground_truth_label}")
    if data_set_name=="mnist":
        verifier=get_mnist_verifier()
        if ground_truth_label not in list(range(10)):
            raise ValueError(f"Ground truth label for mnist should be between 0 and 9, got {ground_truth_label}")
    
    epsilon=max_epsilon_robustness(input_tensor,nap,predicted_label,verifier)
    return {
        "nap": nap,
        "data": input_tensor,
        "label": ground_truth_label,
        "predicted_label":predicted_label,
        "is_correct_class": predicted_label==ground_truth_label,
        "epsilon": input["epsilon"],
        "model": model,
        "data_set": data_set_name,
    }


def real_heuristic_explain_input(
    input_data: ExplanationInput,
    heuristic_name: str,
) -> HeuristicResult:
    
    if data_set_name == "mnist":
        verifier = get_mnist_verifier(input_data["model"])
    elif data_set_name == "breast_cancer":
        verifier = get_breast_cancer_verifier(input_data["model"])
    else:
        raise ValueError(f"Data set not supported: {input_data[data_set_name]}")
    # Now I can use the verifier to get the prediction and the ground truth label
    # start the timer
    t0 = time.perf_counter()
    try:
        coarsened_nap,timeout_flags=coarsen_heuristic(
            nap=input_data["nap"],
            input=input_data["data"],
            label=input_data["label"],
            predicted_label=input_data["predicted_label"],
            epsilon=input_data["epsilon"],
            heuristic_name=heuristic_name,
            verifier=verifier,
            model_path=input_data["model"]
        )
    except Exception as e:
        dt = 0.0
        return HeuristicResult(
            heuristic_name=heuristic_name,
            input_id=image_id,
            model_name=input_data["model"],
            time_taken=dt,
            num_neurons_kept=0,
            total_neurons=0,
            epsilon=input_data["epsilon"],
            predicted_label=input_data["predicted_label"],
            ground_truth_label=input_data["label"],
            num_timeouts=1,
            success=False,
            other_metrics={"error": str(e)},
        )
    dt = time.perf_counter() - t0
    # predicted label
    timeout_count=sum(1 for flag in timeout_flags.values() if flag)
    res=HeuristicResult(
        heuristic_name=heuristic_name,
        input_id=image_id,
        model_name=input_data["model"],
        time_taken=dt,
        num_neurons_kept=get_num_kept(coarsened_nap),
        total_neurons=get_num_total_neurons(coarsened_nap),
        epsilon=input_data["epsilon"],
        predicted_label=input_data["predicted_label"],
        ground_truth_label=input_data["label"],
        num_timeouts=timeout_count,
        success=True,
        other_metrics={"note": "demo"},
    )

    
    return res
"""
    Flatening the heuristic metrics .
    Les clés de other_metrics sont préfixées par m_.
"""

def extract_info(result: HeuristicResult) -> Dict[str, Any]:
   
    row = asdict(result)
    other = row.pop("other_metrics", {}) or {}
    for k, v in other.items():
        row[f"m_{k}"] = v
    # métrique utile : ratio gardé
    if result.total_neurons > 0:
        row["kept_ratio"] = result.num_neurons_kept / result.total_neurons
    else:
        row["kept_ratio"] = None
    return row

"""
Runs all heuristics on a given input and returns the list of HeuristicResult.
"""


def run_heuristics_per_input(
    image: Any,
    input_id: int,
    data_set_name: str,
    model: str = default_model,
    heuristics: Iterable[str] = default_heuristics,
) -> List[HeuristicResult]:
  
    results: List[HeuristicResult] = []
    for h in heuristics:
        try:
            r = real_heuristic_explain_input(
                input_data=image,
                image_id=input_id,
                data_set_name=data_set_name,
                model_name=model,
                heuristic_name=h,
            )  
        except Exception as e:
            # log the error and continue
            print(f"Error running heuristic {h} on input {input_id}: {e}")
            r = HeuristicResult(
                heuristic_name=h,
                input_id=input_id,
                model_name=model,
                time_taken=0.0,
                num_neurons_kept=0,
                total_neurons=0,
                epsilon=0.0,
                num_timeouts=1,
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





def generate_image_id(data_set: str, index: int) -> int:
    """
    Génère un ID unique pour l'image.
    """
    if data_set == "mnist":
        return index  # IDs 0-9999
    elif data_set == "breast_cancer":
        return 100 + index  # IDs 100-199
    else:
        raise ValueError(f"Unsupported dataset: {data_set}")



# Use 
# I can add an arg parsing to choose the dataset , the model , the heuristics
# and the image id
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Explain model decisions using heuristics.")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_DATASETS, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, default=default_model, help="Path to the ONNX model")
    parser.add_argument("--heuristics", type=str, nargs="+", default=default_heuristics, help="List of heuristics to use")
    parser.add_argument("--image_id", type=int, required=True, help="ID of the image to explain")
    return parser.parse_args()




"""
the results folder will be organized like follows :
results/dataset_name_image_n°i_explanations/
    - heuristics_results.csv
    - image.png
    - nap.txt
    - explanation_heuristic1.txt
    - explanation_heuristic2.txt
"""
############################Functions to generate the image related path  ###########################

def get_full_filename(base_path: str, dataset: str,image_idx:int, filename: str) -> str:
    import os
    result_folder= f"{dataset}_image_n°{image_idx}_explanations"
    path = os.path.join(base_path,result_folder)
    os.makedirs(path, exist_ok=True)
    full_path=os.path.join(path,filename)
    return full_path
################# Functions to generate paths for different files in the results folder ###########################
def add_csv_to_results_folder(base_path: str, dataset: str,image_idx:int) -> str:
    filename="heuristics_results.csv"
    return get_full_filename(base_path,dataset,image_idx,filename)
def add_image_to_results_folder(base_path: str, dataset: str,image_idx:int) -> str:
    filename="image.png"
    return get_full_filename(base_path,dataset,image_idx,filename)
def add_nap_to_results_folder(base_path: str, dataset: str,image_idx:int) -> str:
    filename="nap.txt"
    return get_full_filename(base_path,dataset,image_idx,filename)
def add_detailed_explanation_to_results_folder(base_path: str, dataset: str,image_idx:int, heuristic_name:str) -> str:
    filename=f"explanation_{heuristic_name}.txt"
    return get_full_filename(base_path,dataset,image_idx,filename)

def add_analysis_to_results_folder(base_path: str, dataset: str,image_idx:int) -> str:
    filename="explanation_analysis.txt"
    return get_full_filename(base_path,dataset,image_idx,filename)
############# Functions that take care of extracting and exporting results###########################

def export_nap_explanations_as_json(results: list[HeuristicResult], path: str):
    import json
    rows: List[Dict[str, Any]] = [extract_info(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)



def export_input(image: Any, path: str):
    return
############################################# Functions that analyzes the running of the 
# multiple heuristics ###########################
def export_explanation_analysis(results: list[HeuristicResult], path: str):
    # compute some analysis
    analysis = {"num_heuristics": len(results), "successful_heuristics": sum(1 for r    in results if r.success)}   
    with open(path, "w", encoding="utf-8") as f:
        for k, v in analysis.items():
            f.write(f"{k}: {v}\n")
    return

########################### A function that calls some of  the above  functions ####################


def log_results(results: List[HeuristicResult],dataset:str,image_idx:int,image:any,base_path:str):
    rows: List[Dict[str, Any]] = [extract_info(r) for r in results]
    try:
        csv_path = export_results_as_csv(rows, add_csv_to_results_folder(base_path,dataset,image_idx))
        print(f"CSV écrit: {path}")
    except ValueError as e:
        print(f"Export csv failed: {e}")
    # save the naps 
    try:
        nap_path=add_nap_to_results_folder(base_path,dataset,image_idx)
        export_nap_explanations_as_json(results,nap_path)
       
    except ValueError as e:
        print(f"Export naps failed: {e}")
    # save the image
    try:
        path=add_image_to_results_folder(base_path,dataset,image_idx)
        export_input(image,path)
        
    except ValueError as e:
        print(f"Export image failed: {e}")
    try:
        analysis_path=add_analysis_to_results_folder(base_path,dataset,image_idx)
        export_explanation_analysis(results,analysis_path)
    except ValueError as e:
        print(f"Export analysis failed: {e}")
    print("All result  files exported")



############################################Explaining the image ##########################################
################################Using all the desired and supported heuristics ##########################################
def explain_input(
    image: Any,
    input_id: int,
    data_set_name: str,
    model: str = default_model,
    heuristics: Iterable[str] = default_heuristics,
) -> List[HeuristicResult]:
    """
    Runs all heuristics on a given input and returns the list of HeuristicResult.
    """
    results=run_heuristics_per_input(
        image=image,
        input_id=input_id,
        data_set_name=data_set_name,
        model=model,
        heuristics=heuristics,
)    
    log_results(results,dataset=data_set_name,image_idx=input_id,image=image,base_path="results")


    






if __name__ == "__main__":
    
    label=0
    image=get_label_input(label=label)
    onnx_path="mnist-10x2.onnx"
    ground_truth_label=label
    data_set_name="mnist"
    input_explanation=get_explanation_input(image,input_index=1,model_path=onnx_path,ground_truth_label=ground_truth_label,data_set_name=data_set_name)

    
    image_id = generate_image_id(data_set_name, image["id"])
    model = onnx_path
    heuristics = default_heuristics 
    explain_input(
        image=image,
        input_id=image_id,
        data_set_name=data_set_name,
        model=model,
        heuristics=heuristics,
    )
