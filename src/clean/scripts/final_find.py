"""
Trouver les images les plus robustes (ε* max) sous contrainte NAP.
+ export optionnel de ces résultats .

Hypothèses:
- `get_mnist_verifier(model_path) has `is_verified_nap(nap, x, label, eps) -> bool`.
- `nap_extraction_from_onnx(model_path, x)` renvoie une NAP compatible avec le vérifieur.
- Les epsilons sont en norme L∞ , et bornés à [0, 1].

Usage (exemple):
    python robust_images_nap_refactor.py --model models/mnist-10x2.onnx --label 1 \
        --initial-eps 0.3 --consider 100 --top-k 10 --export-json results_label1.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from clean.nap_extraction.extract_nap import nap_extraction_from_onnx
from clean.verifier.verifier_base import get_mnist_verifier
from clean.scripts.useful_func import get_predicted_label_from_model
from clean.nap_explanation.explanation_pipeline import  build_explanation_input , explain_input


def load_mnist_by_label(split: str = "test") -> Dict[int, List[torch.Tensor]]:
    """Charge MNIST groupé par label. split ∈ {'train','test'}.
    Les échantillons sont flatten en vecteurs 784 pour matcher les modèles MLP classiques.
    """
    assert split in {"train", "test"}
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])  # 784-vector
    dataset = MNIST(root="./data", train=(split == "train"), transform=transform, download=True)
    label_to_samples: Dict[int, List[torch.Tensor]] = {i: [] for i in range(10)}
    for x, y in dataset:
        label_to_samples[int(y)].append(x)
    return label_to_samples


def collect_correctly_classified(
    model_path: str,
    label: int,
    split: str = "test",
    max_per_label: int = 500,
) -> List[torch.Tensor]:
    """Garde seulement les images du label prédites correctement par le modèle."""
    buckets = load_mnist_by_label(split=split)
    xs = buckets[label][:max_per_label]
    kept: List[torch.Tensor] = []
    with torch.no_grad():
        for x in xs:
            if get_predicted_label_from_model(model_path, x) == label:
                kept.append(x)
    return kept

# -----------------------------------------------------------------------------
# Coeur de la recherche: calibration d'un ε global puis raffinement par binaire
# -----------------------------------------------------------------------------

@dataclass
class RobustResult:
    index_local: int  # index local dans l'échantillon considéré
    eps_star: float   # ε* (max vérifié sous NAP)
    label: int


def find_robust_images(
    test_set: Sequence[torch.Tensor],
    label: int,
    model_path: str,
    initial_epsilon: float = 0.2,
    number_to_consider: int = 100,
    target_band: Tuple[float, float] = (0.03, 0.2),
    max_global_steps: int = 20,
    top_k: int = 10,
    tol: float = 1e-3,
    max_bin_iters: int = 22,
    seed: int = 42,
) -> Tuple[List[RobustResult], float, List[torch.Tensor]]:
    """
    Renvoie:
      - refined: liste de RobustResult triée desc. par eps_star
      - eps_used: epsilon global utilisé pour le pré-filtrage
      - xs: sous-échantillon d'images effectivement testées (longueur ≤ number_to_consider)
    """
    rng = random.Random(seed)

    # échantillon (shuffle déterministe)
    xs = list(test_set)
    rng.shuffle(xs)
    xs = xs[:number_to_consider]

    verifier = get_mnist_verifier(model_path=model_path)

    # Cache NAPs pour éviter recompute
    naps: List[object] = [None] * len(xs)

    def get_nap(i: int):
        if naps[i] is None:
            naps[i] = nap_extraction_from_onnx(model_path, xs[i])
        return naps[i]

    def ok_at_eps(i: int, eps: float) -> bool:
        nap = get_nap(i)
        return bool(verifier.is_verified_nap(nap, xs[i], label, float(max(0.0, eps))))

    # 1) Ajustement de ε global pour atteindre la fenêtre de taux de succès désirée
    eps = float(max(0.0, initial_epsilon))
    lo_band, hi_band = target_band
    for _ in range(max_global_steps):
        if not xs:
            break
        res = [ok_at_eps(i, eps) for i in range(len(xs))]
        frac = sum(res) / len(xs)
        if lo_band <= frac <= hi_band:
            break
        # trop facile => on augmente ; trop dur => on diminue
        eps = eps * 1.6 if frac > hi_band else eps * 0.6
        eps = max(eps, 0.0)

    # 2) Sélection des candidats à raffiner
    res = [ok_at_eps(i, eps) for i in range(len(xs))]
    pos = [i for i, r in enumerate(res) if r]
    neg = [i for i, r in enumerate(res) if not r]
    to_refine = (pos + neg)[:top_k] if pos else neg[:top_k]

    # 3) Recherche binaire pour ε* (robustesse maximale)
    def max_eps_binary(i: int, hint: float) -> float:
        # borne dure: si ε = 0 ne passe pas, ε* = 0
        if not ok_at_eps(i, 0.0):
            return 0.0

        # Etendre le bracket vers le haut si hint passe déjà
        if ok_at_eps(i, hint):
            lo_e, hi_e = max(0.0, hint), max(hint * 2.0, 1e-4)
            for _ in range(20):
                if not ok_at_eps(i, hi_e):
                    break
                lo_e, hi_e = hi_e, hi_e * 2.0
        else:
            # bracket [0, hint]
            lo_e, hi_e = 0.0, max(hint, 1e-6)

        best = lo_e
        for _ in range(max_bin_iters):
            if hi_e - lo_e <= tol:
                break
            mid = 0.5 * (lo_e + hi_e)
            if ok_at_eps(i, mid):
                best, lo_e = mid, mid
            else:
                hi_e = mid
        return float(best)

    refined: List[RobustResult] = [
        RobustResult(index_local=i, eps_star=max_eps_binary(i, eps), label=label)
        for i in to_refine
    ]
    refined.sort(key=lambda r: r.eps_star, reverse=True)

    return refined, eps, xs



def tensor_to_list(x: torch.Tensor) -> List[float]:
    return x.detach().cpu().view(-1).tolist()


def export_results_json(
    path: Path,
    results: List[RobustResult],
    xs: Sequence[torch.Tensor],
    global_eps: float,
    model_path: str,
):
    """Sauvegarde un JSON auto-portant pour réutilisation/explication ultérieure."""
    payload = {
        "model_path": model_path,
        "global_eps": float(global_eps),
        "count": len(results),
        "items": [
            {
                **asdict(r),
                "input_vector": tensor_to_list(xs[r.index_local]),
            }
            for r in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/mnist-10x2.onnx")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--max-per-label", type=int, default=1000)
    parser.add_argument("--initial-eps", type=float, default=0.3)
    parser.add_argument("--consider", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max-bin-iters", type=int, default=22)
    parser.add_argument("--target-lo", type=float, default=0.05)
    parser.add_argument("--target-hi", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-json", type=str, default="")

    args = parser.parse_args()

    MODEL_PATH = args.model
    LABEL = int(args.label)

    # 1) images correctement classées du split
    correct_test = collect_correctly_classified(
        model_path=MODEL_PATH, label=LABEL, split=args.split, max_per_label=args.max_per_label
    )
    print(f"[info] images correctement classées pour le label {LABEL}: {len(correct_test)}")

    # 2) cherche les plus robustes sous contrainte NAP
    top, eps_used, xs = find_robust_images(
        test_set=correct_test,
        label=LABEL,
        model_path=MODEL_PATH,
        initial_epsilon=float(args.initial_eps),
        number_to_consider=int(args.consider),
        target_band=(float(args.target_lo), float(args.target_hi)),
        top_k=int(args.top_k),
        tol=float(args.tol),
        max_bin_iters=int(args.max_bin_iters),
        seed=int(args.seed),
    )
    print(f"[résumé] ε global utilisé pour le pré-filtrage: {eps_used:.4f}")

    for rank, r in enumerate(top, 1):
        print(f"#{rank:02d} — index_local={r.index_local:3d} | ε* ≈ {r.eps_star:.4f}")

    
    if args.export_json:
        out_path = Path(args.export_json)
        export_results_json(out_path, top, xs, eps_used, MODEL_PATH)
        print(f"[ok] Résultats exportés → {out_path}")

"""
This function takes the path where we saved results and loads them

"""
# returns list of tensors ,list of their espilon robustness

# I just neet to save them in a good way 
def load_robust_samples(num_robust_desired, path_robust_inputs):

   
  

    return robust_inputs,robust_epsilons




def explain_some_robust_inputs(dataset_name,model_path):
    label=1
    inputs, epsilons=load_robust_samples()
    # Now I am explaining some of the inputs
    # I will call the explanation pipeline here
    for i in range(len(inputs)):
        explanation_input=build_explanation_input(inputs[i], gt_label=label, dataset="mnsit", model_path=model_path)
        # explain it
        explain_input(explanation_input,i, dataset=dataset_name, model=model_path,outputs_base= "outputs_robust")






if __name__ == "__main__":
    main()
    explain_some_robust_inputs()