import torch
import torch.nn as nn
import torch.nn.functional as F

class FMnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)


        self.linear = nn.Linear(512, 10)
        self.feature_extractor = nn.Sequential(self.conv1, nn.ReLU(), self.pool1, self.conv2, nn.ReLU(), self.pool2, nn.Flatten(), 
                                               self.fc1, nn.ReLU())
        self.classifier = nn.Sequential(self.linear)
    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        x = self.feature_extractor(tensor)
        tensor = self.classifier(x)
        return x, tensor
    def get_model_size(self, part='full'):
        total_params = 0
        model_part = self

        # Select part of the model to calculate parameters
        if part == 'classifier':
            model_part = self.classifier

        # Calculate parameters
        for name, param in model_part.named_parameters():
            layer_params = param.numel()
            total_params += layer_params

        total_params_mb = (total_params * 4) / 1024 / 1024
        return total_params_mb


    #     self.conv = nn.Sequential(
    #         nn.Conv2d(1, 32, 5),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, stride=2),
    #         #nn.Dropout(0.3),
    #         nn.Conv2d(32, 64, 5),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, stride=2),
    #        # nn.Dropout(0.3)
    #     )
    #     self.fc = nn.Sequential(
    #         nn.Linear(64*4*4, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 10)
    #     )
        
    # def forward(self, x):
    #     x = self.conv(x)
    #     x = x.view(-1, 64*4*4)
    #     x = self.fc(x)
    #     # x = nn.functional.normalize(x)
    #     return x
if __name__ == '__main__':
    model = FMnist_CNN()
    
    # Calculate and print the size of the full model
    total_model_size = model.get_model_size('full')
    print(f"Total model size: {total_model_size:.2f} MB")
    
    # Calculate and print the size of the classifier part
    classifier_size = model.get_model_size('classifier')
    print(f"Classifier size: {classifier_size:.2f} MB")