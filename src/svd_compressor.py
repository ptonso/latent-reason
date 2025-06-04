import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from tqdm import tqdm
import copy
import time


from src.logger import setup_logger


logger = setup_logger("tucker_eval.log")



def get_dataloader(data_root: str, batch_size: int = 128, num_workers: int = 4):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(root=f"{data_root}/train", transform=transform_train)
    val_set = datasets.ImageFolder(root=f"{data_root}/val", transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader



def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def fine_tune(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device,
              epochs: int = 5, lr: float = 0.01):
    model.train()
    optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        logger.info(f"Fine-tune Epoch {epoch+1}, Accuracy: {acc:.4f}")


def compress_model(model: nn.Module, ratio: float = 0.2) -> nn.Module:
    compressor = TuckerCompressor()

    def compress_layer(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                r_out = max(1, int(child.out_channels * ratio))
                r_in = max(1, int(child.in_channels * ratio))
                setattr(module, name, compressor.compress_conv2d(child, rank=(r_out, r_in)))
            else:
                compress_layer(child)

    model = copy.deepcopy(model)
    compress_layer(model)
    return model



def get_resnet18_100_classes():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 100)  # 100 CIFAR-100 classes
    return model
