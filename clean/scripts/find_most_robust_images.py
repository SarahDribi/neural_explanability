# robust_images_nap.py
# Trouver les images MNIST les plus robustes (ε* max) sous contrainte NAP.

import random
from collections import defaultdict
from typing import List, Tuple

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda

from clean.nap_extraction.extract_nap import nap_extraction_from_onnx
from clean.verifier.verifier_base import get_mnist_verifier
from clean.verifier.verifier_base import get_breast_cancer_verifier
from clean.scripts.useful_func import get_predicted_label_from_model
from clean.scripts.useful_func import max_epsilon_nap_robustness
#(x, label, verifier, low=0.001, high=1, tol=0.001, max_iter=10)


def load_mnist_by_label(split: str = "test") -> dict:
    """Charge MNIST groupé par label. split ∈ {'train','test'}."""
    assert split in {"train", "test"}
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])  # 784-vector
    dataset = MNIST(root="./data", train=(split == "train"), transform=transform, download=True)
    label_to_samples = defaultdict(list)
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
    kept = []
    for x in xs:
        if get_predicted_label_from_model(model_path, x) == label:
            kept.append(x)
    return kept


def find_robust_images(
    test_set: List[torch.Tensor],
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
):
    """
    Renvoie:
      - refined: liste [(index_local, eps_star), ...] triée desc.
      - eps_used: epsilon global utilisé pour le pré-filtrage.
    """
    random.seed(seed)

    # take a random sample
    xs = list(test_set)
    random.shuffle(xs)
    xs = xs[:number_to_consider]

    
    verifier = get_mnist_verifier(model_path=model_path)

    # Caching  NAPs
    naps = [None] * len(xs)

    def get_nap(i):
        if naps[i] is None:
            naps[i] = nap_extraction_from_onnx(model_path, xs[i])
        return naps[i]

    def ok_at_eps(i, eps: float) -> bool:
        nap = get_nap(i)
        
        return bool(verifier.is_verified_nap(nap, xs[i], label, eps))

    # ajustement de  ε global pour obtenir une fraction de succès dans la bande cible
    eps = float(initial_epsilon)
    lo, hi = target_band
    for _ in range(max_global_steps):
        res = [ok_at_eps(i, eps) for i in range(len(xs))]
        frac = sum(res) / max(1, len(xs))
        if lo <= frac <= hi:
            break
        # trop facile => on augmente ; trop dur => on diminue
        eps = eps * 1.6 if frac > hi else eps * 0.6
        
        eps = max(eps, 0.0)

    # 2) sélectionner des candidats à raffiner : d’abord ceux qui passent à eps
    res = [ok_at_eps(i, eps) for i in range(len(xs))]
    pos = [i for i, r in enumerate(res) if r]  # ceux qui passent 
    neg = [i for i, r in enumerate(res) if not r] # ceux qui ne passent pas
    to_refine = (pos + neg)[:top_k] if pos else neg[:top_k]  # s'il n'y a aucun pass

    # 3) recherche binaire pour ε* (robustesse maximale)
    def max_eps_binary(i: int, hint: float) -> float:
        # si ε = 0 ne passe pas, ε* = 0
        if not ok_at_eps(i, 0.0):
            return 0.0

        # étendre le bracket vers le haut si hint passe déjà
        if ok_at_eps(i, hint):
            lo_e, hi_e = hint, max(hint * 2.0, 1e-4)
            for _ in range(16):
                if not ok_at_eps(i, hi_e):
                    break
                lo_e, hi_e = hi_e, hi_e * 2.0
        else:
            # sinon bracket [0, hint] en repliant hi vers le bas
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
    
    refined = [(i, max_eps_binary(i, eps)) for i in to_refine]
    refined.sort(key=lambda t: t[1], reverse=True)

    return refined, eps,xs


if __name__ == "__main__":
    # paramètres principaux
    MODEL_PATH = "models/mnist-10x2.onnx"
    LABEL = 1
    INITIAL_EPS = 0.3
    verifier = get_mnist_verifier(model_path=MODEL_PATH)

    # 1) on prend des images correctement classées du *test set*
    correct_test = collect_correctly_classified(
        model_path=MODEL_PATH, label=LABEL, split="test", max_per_label=1000
    )
    print(f"[info] images correctement classées pour le label {LABEL}: {len(correct_test)}")

    # 2) on cherche les plus robustes
    top, eps_used ,xs= find_robust_images(
        test_set=correct_test,
        label=LABEL,
        model_path=MODEL_PATH,
        initial_epsilon=INITIAL_EPS,
        number_to_consider=10,   # ⇐ échantillon de travail
        target_band=(0.05, 0.2),  # ⇐ mélange pass/fail souhaité quand on calibre ε global
        top_k=10,
        tol=1e-3,
        max_bin_iters=5,
        seed=42,
    )
    # for all the max robust images find their local robustness
    print(f"[résumé] ε global utilisé pour le pré-filtrage: {eps_used:.4f}")

    
    


        #print(f"#{rank:02d} — index_local={idx:3d} | ε* ≈ {eps_star:.4f}")

    for rank, (idx, eps_star) in enumerate(top, 1):
      
        print(f"#{rank:02d} — index_local={idx:3d} | ε* ≈ {eps_star:.4f} ")

# find most_robust_images 


# I need to rewrite the algorithm 
# First prefilter with a global epsilon
#it should be easy for some samples and hard for others
# then run the binary verification on the other images 
# for a limited time 

# je dois aussi écrire le coarsening de l'un
# I will also compare models
#compare across models