
from typing import Tuple, Optional

import torch
from src.explanation_logic.dataset.other_dataset import load_data_set, split_then_normalize




class TestSetLoader:
    """
   
      - WBC:   tensor (1,9)        normalisé comme dans split_then_normalize
    """
    def __init__(self, dataset: str):
        self.dataset = dataset
        ds = (dataset or "").lower().strip()

        self._Xte: Optional[torch.Tensor] = None
        self._yte: Optional[torch.Tensor] = None
        if ds == "breast_cancer":
            # Ces fonctions viennent de src.explanation_logic.dataset.other_dataset
            X, y = load_data_set()
            # Ici, return_scaler=False => la fonction doit renvoyer des TENSEURS
            _, _, X_te, y_te = split_then_normalize(X, y, test_size=0.2, seed=42, return_scaler=False)
            # Pas de conversion numpy -> torch: ce sont déjà des tensors
            self._Xte = X_te  # float32, shape (N, 9)
            self._yte = y_te.long()  # int64, shape (N,)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset!r}. Use 'mnist' or 'breast_cancer'.")

    def __len__(self) -> int:
        if self._mnist is not None:
            return len(self._mnist)
        if self._Xte is not None:
            return int(self._Xte.shape[0])
        return 0

    def get(self, idx: int) -> Tuple[torch.Tensor, int]:
      
            # WBC
            x = self._Xte[idx].unsqueeze(0) # -> (1,9)
            y = int(self._yte[idx].item())
            return x, y

   

if __name__ == "__main__":
    # smoke tests rapides (exécuter ce fichier directement pour vérifier)
    print("=== Smoke test WBC ===")
    wbc_loader = TestSetLoader("breast_cancer")
    x, y = wbc_loader.get(0)
    print("WBC shapes:", x.shape, type(y), y)

    print("=== Smoke test MNIST ===")
    mnist_loader = TestSetLoader("mnist")
    x, y = mnist_loader.get(0)
    print("MNIST shapes:", x.shape, type(y), y)
