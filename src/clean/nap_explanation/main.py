import argparse, logging
from clean.nap_explanation.explanation_types import SUPPORTED_DATASETS
from clean.nap_explanation.explanation_pipeline import build_explanation_input, explain_input
from clean.dataset.loader import DatasetLoader

DEFAULT_HEURISTICS=["simple","random"]

def parse_args():
    p = argparse.ArgumentParser("napx")
    p.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="mnist")
    p.add_argument("--model", default="models/mnist-10x2.onnx")
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--tag", default="")
    p.add_argument("--label", type=int, help="optional single class label")
    # i also need to add the heuristics list to the parser
    p.add_argument('--heuristics', type=str, nargs='+', default=DEFAULT_HEURISTICS,
                   help='List of heuristics to test (default: simple random).')
    p.add_argument("--coarsening_timeout_step", type=int, default=10)
    
    return p.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    delta=0.05

    labels = [args.label] if args.label is not None else ([1] if args.dataset=="mnist" else [1,0])
    for label in labels:
        loader = DatasetLoader(dataset_name="mnist",num_classes=10) if args.dataset=="mnist" else DatasetLoader(dataset_name="breast_cancer",num_classes=2)
        inputs = loader.get_label_samples(label=label, num_samples=args.num_images)
        for idx, image in enumerate(inputs):
            exp = build_explanation_input(image, gt_label=label, dataset=args.dataset, model_path=args.model)
            if exp.epsilon<exp.epsilon_region+delta:
                logging.info("Skip image %d:  epsilon nap not so bigger", idx); continue
            _ = explain_input(exp, input_id=label + idx*10, dataset=args.dataset, model=args.model, heuristics =args.heuristics,
                              outputs_base=args.outputs, tag=f"{args.dataset}_img{label}_{idx}",coarsening_timeout_step=args.coarsening_timeout_step)

if __name__ == "__main__":
    main()
