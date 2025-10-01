from src.dataset.mnist_data import  TestLoaderMnist  
from src.explanation_logic.explanation_runner import ExplanationRunner

runner = ExplanationRunner(model="models/mnist-10x2.onnx", dataset="mnist")
runner.set_outputs_base("new_outputs").set_coarsening_timeout_step(20)

# charger le test set
loader = TestLoaderMnist()  # ou "breast_cancer"

# expliquer le 0ème échantillon
results = runner.explain_index(dataloader=loader, idx=0, tag="demo")