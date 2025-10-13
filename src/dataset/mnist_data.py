import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Lambda

class TestLoaderMnist:
    def __init__(self, batch_size=64, flatten=True, seed: int = 42):
        if flatten:
            transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
        else:
            transform = ToTensor()
        
        self.dataset = MNIST(
            root="./data",
            train=False,
            transform=transform,
            download=True
        )
        
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,   
            generator=g     
        )
    
    def get_loader(self):
        return self.loader
    
    def get_test_sample(self, i: int):
        """Retourne (image, label) pour l'échantillon i du dataset test"""
        return self.dataset[i]

testloader = TestLoaderMnist(batch_size=128, flatten=False, seed=123)

# Toujours le même ordre à chaque run
for imgs, labels in testloader.get_loader():
    print(labels[:10])  
    break


sample= testloader.get_test_sample(42)
print("Sample:", sample)





