from typing import Tuple, Optional

import torch
from src.explanation_logic.dataset.other_dataset import load_data_set, split_then_normalize


class TestSetLoader:
    """
    Charge le jeu de test pour le dataset Breast Cancer.
    get(idx) -> (input_tensor_batched, gt_label)
      - WBC: tensor (1,9) normalisé comme dans split_then_normalize
    """
    def __init__(self, dataset: str):
        ds = (dataset or "").lower().strip()
        if ds == "breast_cancer":
            X, y = load_data_set()
            _, _, X_te, y_te = split_then_normalize(X, y, test_size=0.2, seed=42, return_scaler=False)
            self._Xte: torch.Tensor = X_te  # float32, shape (N,9)
            self._yte: torch.Tensor = y_te.long()  # int64, shape (N,)
        else:
            raise ValueError(f"Unsupported dataset: {dataset!r}. Use 'breast_cancer'.")

    def __len__(self) -> int:
        return int(self._Xte.shape[0])

    def get(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self._Xte[idx].unsqueeze(0) 
        y = int(self._yte[idx].item())
        return x, y


if __name__ == "__main__":
    print("=== Smoke test WBC ===")
    wbc_loader = TestSetLoader("breast_cancer")
    for i  in range(3):
        x, y = wbc_loader.get(i)
        print("WBC shapes:", x.shape, type(y), y)
        print(x)
        print(y)
