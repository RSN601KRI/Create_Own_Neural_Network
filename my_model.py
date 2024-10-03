import torch
from torchvision import models
from torch import nn, optim

def build_vgg16(hidden):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden, 102),
        nn.LogSoftmax(dim=1)
    )
    return model

def build_resnet50(hidden):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, hidden),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden, 102),
        nn.LogSoftmax(dim=1)
    )
    return model

def create_nn_model(model_name, hidden, lr, gpu=False):
    model = None

    print(f"Model Name: {model_name}")
    print(f"Hidden Units: {hidden}")
    print(f"Learning Rate: {lr}")

    # Using factory functions to build models
    if model_name == 'vgg16':
        model = build_vgg16(hidden)
    elif model_name == 'resnet50':
        model = build_resnet50(hidden)
    else:
        print("No such model, select either 'vgg16' or 'resnet50'.")
        exit()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    if gpu:
        model = model.to("cuda")
        criterion = criterion.to("cuda")

    return model, criterion, optimizer
