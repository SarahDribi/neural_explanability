from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import torch

from src.explanation_logic.nap_explanation.explanation_pipeline import build_explanation_input, explain_input
from src.explanation_logic.nap_explanation.explanation_types import (
    DEFAULT_HEURISTICS, ExplanationInput, HeuristicResult
)


class ExplanationRunner:
    """
    Wrapper  autour de l'explanation_pipeline.
    - model: chemin vers le modèle (ex: 'models/mnist.onnx' ou 'artifacts/wbc_mlp.onnx')
    - dataset: 'mnist' ou 'breast_cancer' (ou autre si tu fournis un dataloader adapté)
    - heuristics: liste des heuristiques à appliquer (par défaut: DEFAULT_HEURISTICS)
    - coarsening_timeout_step: pas de timeout 
    """

    def __init__(
        self,
        model: str,
        dataset: str,
        outputs_base: str = "outputs",
        heuristics: Optional[Iterable[str]] = None,
        coarsening_timeout_step: int = 12,
    ):
        self.model = model
        self.dataset = dataset
        self.outputs_base = outputs_base
        self.heuristics = list(heuristics) if heuristics is not None else list(DEFAULT_HEURISTICS)
        self.coarsening_timeout_step = coarsening_timeout_step

        # s'assure que le dossier de sortie existe
        Path(self.outputs_base).mkdir(parents=True, exist_ok=True)

    # ---------- Setters ----------
    def set_model(self, model: str):
        self.model = model
        return self

    def set_dataset(self, dataset: str):
        self.dataset = dataset
        return self

    def set_outputs_base(self, outputs_base: str):
        self.outputs_base = outputs_base
        Path(self.outputs_base).mkdir(parents=True, exist_ok=True)
        return self

    def set_heuristics(self, heuristics: Iterable[str]):
        self.heuristics = list(heuristics)
        return self

    def add_heuristic(self, heuristic: str):
        if heuristic not in self.heuristics:
            self.heuristics.append(heuristic)
        return self

    def remove_heuristic(self, heuristic: str):
        if heuristic in self.heuristics:
            self.heuristics.remove(heuristic)
        return self

    def set_coarsening_timeout_step(self, step: int):
        self.coarsening_timeout_step = step
        return self

    
    def _get_test_input_by_index(self, dataloader, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Attend un objet 'dataloader' qui expose une méthode:
            get(idx) -> (input_tensor_batched, gt_label)
        ou bien
            get_test_sample(idx) -> (input_tensor_batched, gt_label)

        Pour MNIST/WBC, tu peux utiliser ton TestSetLoader.
        """
        # Cas A: API type TestSetLoader (get)
        if hasattr(dataloader, "get"):
            return dataloader.get(idx)
        # Cas B: API alternative (get_test_sample)
        if hasattr(dataloader, "get_test_sample"):
            return dataloader.get_test_sample(idx)
        raise NotImplementedError(
            "Le dataloader doit définir 'get(idx)' ou 'get_test_sample(idx)'."
        )


    def explain(
        self,
        input_tensor: torch.Tensor,
        gt_label: int,
        input_id: int,
        tag: str = "",
    ) -> List[HeuristicResult]:
        """
        Construit un ExplanationInput et exécute les heuristiques.
        Retour: List[HeuristicResult]
        """
        exp: ExplanationInput = build_explanation_input(
            input_tensor=input_tensor,
            gt_label=gt_label,
            dataset=self.dataset,
            model_path=self.model,
        )

        results: List[HeuristicResult] = explain_input(
            exp=exp,
            input_id=input_id,
            dataset=self.dataset,
            model=self.model,
            heuristics=self.heuristics,
            outputs_base=self.outputs_base,
            tag=tag,
            coarsening_timeout_step=self.coarsening_timeout_step,
        )
        return results

    def explain_index(
        self,
        dataloader,
        idx: int,
        tag: str = "",
    ) -> List[HeuristicResult]:
        """
        Va chercher le i-ème échantillon du test set via 'dataloader',
        puis appelle 'explain' avec les bons paramètres.
        """
        input_tensor, gt_label = self._get_test_input_by_index(dataloader, idx)
        
        #import pdb; pdb.set_trace()
        return self.explain(
            input_tensor=input_tensor,
            gt_label=int(gt_label),
            input_id=int(idx),
            tag=tag,
        )


