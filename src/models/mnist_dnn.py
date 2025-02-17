import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.linear = nn.Linear(128, 10)
        self.feature_extractor = nn.Sequential(self.layer1, nn.ReLU(), self.layer2, nn.ReLU(), self.layer3, nn.ReLU())
        self.classifier = nn.Sequential(self.linear)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.feature_extractor(x)
        tensor = self.classifier(x)
        return x, tensor

    def get_model_size(self, ):
        total_params = 0
        for name, param in Mnist_DNN().named_parameters():
            layer_params = param.numel()
            total_params += layer_params
            # print(f"{name}: {layer_params} parameters")
        total_params_kb = (total_params * 4) / 1024 / 1024
        return total_params_kb


if __name__ == '__main__':
    # Calculate and print parameters for each layer
    total_params = 0
    for name, param in Mnist_DNN().named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")
    total_params_kb = (total_params * 4) / 1024 / 1024
    print(f"Total model parameters: {total_params_kb:.2f} MB")