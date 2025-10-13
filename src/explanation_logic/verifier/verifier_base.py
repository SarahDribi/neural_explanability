from src.explanation_logic.verifier.nap_constrained_verif import  verify_nap_property_around_input
from src.explanation_logic.verifier.region_robustness_verif import verify_robustness_around_input

from src.explanation_logic.verifier.region_robustness_verif import branch_and_bound_verif

class Verifier:
    def __init__(self, model_path, json_config, use_gpu, timeout,num_classes,timeout_coarsening_step=40):
        self.model_path = model_path

        self.json_config = json_config
        self.timeout = timeout
        self.use_gpu = use_gpu
        self.num_classes=num_classes
        self.timeout_coarsening_step=timeout_coarsening_step
        

    def is_verified_nap(self, nap,input, label, epsilon):
        use_gpu=self.use_gpu
        result,timed_out = verify_nap_property_around_input(
            model_path=self.model_path,
            input=input,
            epsilon=epsilon,
            json_config=self.json_config,
            
            nap=nap,
            label=label,
            use_gpu=use_gpu,
            timeout=self.timeout,
            num_classes=self.num_classes
            
    
        )
        return result 
    def is_verified_nap_small_timeout(self, nap, input, label, epsilon):
        use_gpu=self.use_gpu
        result, timed_out = verify_nap_property_around_input(
            model_path=self.model_path,
            input=input,
            epsilon=epsilon,
            json_config=self.json_config,
            nap=nap,
            label=label,
            use_gpu=use_gpu,
            timeout=self.timeout_coarsening_step,
            num_classes=self.num_classes
        )
        return result, timed_out  # return both result and timeout status

    def is_verified_region(self,input,label,epsilon):
        use_gpu=self.use_gpu
        result=verify_robustness_around_input(model_path=self.model_path,input=input,epsilon=epsilon,json_config=self.json_config, label=label, use_gpu=use_gpu,timeout=self.timeout,num_classes=self.num_classes)
        return  result
    def verif_without_milp(self,input,label,epsilon):
        use_gpu=self.use_gpu
        res=branch_and_bound_verif(model_path=self.model_path,input=input,epsilon=epsilon,json_config=self.json_config, label=label, use_gpu=use_gpu,timeout=self.timeout,num_classes=self.num_classes)
        return res
    def get_num_classes(self):
        return self.num_classes





# Define a ready to use verifier 
# A verifier for the Small model
def get_mnist_verifier(timeout=3000,model_path="clean/models/small_model.onnx",json_config="bab_configs/mnistfc_vnncomp21.json",coarsening_timeout_step=12):
    
    use_gpu = False
    num_classes=10
    return Verifier(model_path, json_config, use_gpu, timeout,num_classes,timeout_coarsening_step=coarsening_timeout_step)


def get_breast_cancer_verifier(timeout=3000,model_path="artifacts/wbc_mlp.onnx",json_config="bab_configs/mnistfc_vnncomp21.json",coarsening_timeout_step=12):
    use_gpu = False
    
    return Verifier(model_path, json_config, use_gpu, timeout,num_classes=2,timeout_coarsening_step=coarsening_timeout_step)

