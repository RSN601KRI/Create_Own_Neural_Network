import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from transformer import my_transforms
from my_model import create_nn_model

def setup_device(gpu):
    if gpu and torch.cuda.is_available():
        print("GPU in use.")
        return "cuda"
    else:
        print("CPU in use")
        return "cpu"

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc="Training")

    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))

    return running_loss / len(train_loader)

def validate_model(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    valid_loader_tqdm = tqdm(valid_loader, desc="Validating")

    with torch.no_grad():
        for inputs, labels in valid_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            valid_loader_tqdm.set_postfix(loss=running_loss / len(valid_loader))

    return running_loss / len(valid_loader), 100 * correct / total

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            test_loss += loss.item()
            _, predicted = preds.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return test_loss / len(test_loader), 100 * correct / total

def save_checkpoint(model, save_dir, arch, hidden, epochs, learn_rate, train_data):
    model.class_to_idx = train_data.class_to_idx
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    checkpoint = {
        'architecture': arch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'hidden': hidden,
        'epochs': epochs,
        'learning_rate': learn_rate
    }
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, f"checkpoint-{arch}.pth"))

def train_my_model(data_dir, save_dir, my_arch, hidden, learn_rate, epochs, gpu=False):
    print(f"Model: {my_arch}, Hidden Layers: {hidden}, Learn Rate: {learn_rate}, Epochs: {epochs}")

    device = setup_device(gpu)
    train_loader, valid_loader, test_loader, train_data = my_transforms(
        os.path.join(data_dir, 'train'),
        os.path.join(data_dir, 'valid'),
        os.path.join(data_dir, 'test')
    )

    model, criterion, optimizer = create_nn_model(my_arch, hidden, learn_rate, gpu)

    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, accuracy = validate_model(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.2f}%')

    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.2f}%")

    save_checkpoint(model, save_dir, my_arch, hidden, epochs, learn_rate, train_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network and save as checkpoint.")
    parser.add_argument('data_dir', type=str, help="Specify the path to where the datasets are stored.")
    parser.add_argument('--save_dir', type=str, default="my_checkpoints", help="Specify path to where models are saved.")
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help="Specify which model architecture.")
    parser.add_argument('--hidden', type=int, default=512, help="Specify the hidden layer neurons.")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="Specify the learning rate for the model.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of iterations to train the model.")
    parser.add_argument('--gpu', action='store_true', help="Specify whether to use the gpu or not.")

    args = parser.parse_args()

    print("Arguments Collected")
    train_my_model(args.data_dir, args.save_dir, args.arch, args.hidden, args.learn_rate, args.epochs, args.gpu)
