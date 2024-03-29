import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
import utils
from training_pipeline import train
from tqdm import tqdm


# Getting/Downloading a dataset
train_data = datasets.FashionMNIST(root = "./data", train=True , transform=ToTensor(),download=True,target_transform=None)
test_data = datasets.FashionMNIST(root = "./data", train=False , transform=ToTensor(),download=True)

# Preparing iterable dataloader
train_dataloader = DataLoader(train_data , batch_size=32 , shuffle=True)
test_dataloader = DataLoader(test_data , batch_size=32 , shuffle=False)

if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


# Create a model with non-linear and linear layers
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
    
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784, # number of input features
    hidden_units=10,
    output_shape=len(train_data.classes) # number of output classes desired
).to(device) # send model to GPU if it's available
next(model_1.parameters()).device # check model device

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

train(model_1,train_dataloader,test_dataloader,loss_fn,optimizer,utils.accuracy_fn,device,3)
utils.eval(model_1,test_dataloader,loss_fn,utils.accuracy_fn,device)

utils.save_model(model_1,"models","FMNIST_relu.pth")