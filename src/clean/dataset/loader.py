from clean.dataset.utils import get_label_inputs
from clean.dataset.other_dataset import get_label_inputs_breast_cancer

class DatasetLoader:
    def __init__(self, dataset_name, num_classes):
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.loaders = {
            "mnist": get_label_inputs,
            "breast_cancer": get_label_inputs_breast_cancer,
        }

    def get_label_samples(self, label, num_samples):
        if self.dataset_name not in self.loaders:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        return self.loaders[self.dataset_name](label=label, num_samples=num_samples)

    
    def check_compatibility(model_path):
        return True
