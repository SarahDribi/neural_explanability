from src.dataset.mnist_data import  TestLoaderMnist  
from src.explanation_logic.explanation_runner import ExplanationRunner
from src.dataset.breast_cancer_data import TestLoaderWBC
#The testloaders should expose this method get_test_sample(i)
if __name__ == "__main__":
    # MNIST
    
    runner = ExplanationRunner(model="models/mnist-10x2.onnx", dataset="mnist")
    runner.set_outputs_base("new_outputs").set_coarsening_timeout_step(20)
    loader = TestLoaderMnist()  
    results = runner.explain_index(dataloader=loader, idx=0, tag="demo_mnist")
    
    # Breast cancer

    
    runner_bc = ExplanationRunner(model="artifacts/wbc_mlp.onnx", dataset="breast_cancer").set_coarsening_timeout_step(20)
    loader_bc = TestLoaderWBC(batch_size=128, seed=123)
    results_bc = runner_bc.explain_index(dataloader=loader_bc, idx=0, tag="demo_wbc")
    