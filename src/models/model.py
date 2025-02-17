from src.models.cifar10_alexnet import CIFAR10_AlexNet
from src.models.mnist_dnn import Mnist_DNN
from src.models.fmnist_cnn import FMnist_CNN
from src.models.emnist_cnn import EMNISTCNN
from src.models.imagenette2_resnet import WideResNet

# from src.models.tiny_imagenet_resnet import ResNet34
import torch
import torch.nn as nn
import numpy as np


def choose_model(options):
    model_name = str(options['model_name']).lower()
    torch.manual_seed(2030)
    if model_name == 'mnist_dnn':
        # for name, param in Mnist_CNN().named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        #         break
        return Mnist_DNN()
    elif model_name == 'fmnist_cnn':
        return FMnist_CNN()
    elif model_name == 'emnist_cnn':
        return EMNISTCNN()
    elif model_name == 'cifar10_alexnet':
        return CIFAR10_AlexNet()
    elif model_name == 'imagenette2_resnet':
        return WideResNet()