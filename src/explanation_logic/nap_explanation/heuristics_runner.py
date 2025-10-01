
import time, logging
from typing import Iterable, List
from src.explanation_logic.nap_explanation.explanation_types import HeuristicResult, ExplanationInput, count_timeouts
from src.explanation_logic.coarsening.nap_coarsen import coarsen_heuristic
from src.explanation_logic.nap_extraction.nap_utils import get_num_kept, get_num_total_neurons
from src.explanation_logic.nap_explanation.useful_func import max_epsilon_robustness
from src.explanation_logic.nap_explanation.verifier import get_verifier

log = logging.getLogger(__name__)

def run_one_heuristic(
    exp: ExplanationInput, input_id: int, data_set_name: str, model_name: str, heuristic_name: str
,coarsening_timeout_step=12) -> HeuristicResult:
    verifier = get_verifier(data_set_name, model_name,coarsening_timeout_step=coarsening_timeout_step)
    t0 = time.perf_counter()
    try:
        coarsened, timeout_flags = coarsen_heuristic(
            nap=exp.nap,
            input=exp.data,
            label=exp.predicted_label,
            epsilon=exp.epsilon,
            heuristic_name=heuristic_name,
            verifier=verifier,
            model_path=exp.model,
        )
        post_eps = max_epsilon_robustness(exp.data, coarsened, exp.predicted_label, verifier)
        log.debug("epsilon after coarsening (%s): %.6f (before: %.6f)",
                  heuristic_name, post_eps, exp.epsilon)
        dt = time.perf_counter() - t0
        return HeuristicResult(
            heuristic_name=heuristic_name,
            input_id=input_id,
            model_name=model_name,
            time_taken=dt,
            num_neurons_kept=get_num_kept(coarsened),
            total_neurons=get_num_total_neurons(coarsened),
            epsilon=exp.epsilon,
            epsilon_region=exp.epsilon_region,
            predicted_label=exp.predicted_label,
            ground_truth_label=exp.label,
            num_timeouts=count_timeouts(timeout_flags),
            coarsened_nap=coarsened,
            success=True,
            other_metrics={"epsilon_after": post_eps},
        )
    except Exception as e:
        dt = time.perf_counter() - t0
        log.exception("Heuristic %s failed on input %d", heuristic_name, input_id)
        return HeuristicResult(
            heuristic_name=heuristic_name,
            input_id=input_id,
            model_name=model_name,
            time_taken=dt,
            num_neurons_kept=0,
            total_neurons=0,
            epsilon=exp.epsilon,
            epsilon_region=-2.0,
            predicted_label=exp.predicted_label,
            ground_truth_label=exp.label,
            num_timeouts=1,
            coarsened_nap=None,
            success=False,
            other_metrics={"error": str(e)},
        )

def run_many_heuristics(exp: ExplanationInput, input_id: int, dataset: str, model: str, heuristics: Iterable[str],coarsening_timeout_step=12) -> List[HeuristicResult]:
    return [run_one_heuristic(exp, input_id, dataset, model, h,coarsening_timeout_step=coarsening_timeout_step) for h in heuristics]
