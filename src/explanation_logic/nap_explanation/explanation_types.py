
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

SUPPORTED_DATASETS = {"mnist", "breast_cancer"}
DEFAULT_HEURISTICS: List[str] = [
    "simple"
]

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
    coarsened_nap: Optional[List[List[int]]]
    success: bool
    other_metrics: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        other = row.pop("other_metrics") or {}
        for k, v in other.items():
            row[f"m_{k}"] = v
        row["kept_ratio"] = (
            self.num_neurons_kept / self.total_neurons if self.total_neurons else None
        )
        return row

@dataclass
class ExplanationInput:
    nap: List[List[int]]
    data: Any
    label: int
    predicted_label: int
    is_correct_class: bool
    epsilon: float
    epsilon_region: float
    model: str
    data_set: str
    region_constrained_activations: List[List[int]]

def count_timeouts(flags) -> int:
    if flags is None: return 0
    if isinstance(flags, (list, tuple, set)):
        return sum(count_timeouts(v) for v in flags)
    if isinstance(flags, dict):
        return sum(count_timeouts(v) for v in flags.values())
    return int(bool(flags))
