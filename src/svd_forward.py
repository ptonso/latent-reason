import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import logging
from typing import Tuple, List

from src.logger import setup_logger

class TrainingConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    svd_rank: int = 64
    finetune_epochs: int = 3  # additional epochs for fine-tuning after compression
    benchmark_batches: int = 100 
    

class FullNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FullNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CompressedNet(nn.Module):
    def __init__(
        self,
        fc1a: nn.Linear,
        fc1b: nn.Linear,
        fc2: nn.Linear,
        fc3: nn.Linear,
    ) -> None:
        super().__init__()
        # Compressed version of fc1 is implemented as two sequential layers.
        self.fc1a = fc1a
        self.fc1b = fc1b
        self.fc2 = fc2
        self.fc3 = fc3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1a(x)
        x = self.fc1b(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compress_fc_layer(fc_layer: nn.Linear, k: int) -> Tuple[nn.Linear, nn.Linear]:
    """
    Compress a fully connected layer using SVD.
    Returns two linear layers that together approximate the original.
    """
    try:
        with torch.no_grad():
            # fc_layer.weight shape: [out_features, in_features]
            W = fc_layer.weight.data
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            # Explicitly sort singular values in descending order.
            indices = torch.argsort(S, descending=True)
            U = U[:, indices]
            S = S[indices]
            Vt = Vt[indices, :]

            U_k = U[:, :k]   # [out_features, k]
            S_k = S[:k]      # [k]
            Vt_k = Vt[:k, :] # [k, in_features]

        fc1a = nn.Linear(fc_layer.in_features, k, bias=False)
        fc1a.weight.data.copy_(Vt_k)

        fc1b = nn.Linear(k, fc_layer.out_features, bias=True)
        weight_fc1b = U_k * S_k.unsqueeze(0)  # Broadcasting S_k over columns.
        fc1b.weight.data.copy_(weight_fc1b)
        if fc_layer.bias is not None:
            fc1b.bias.data.copy_(fc_layer.bias.data)
        return fc1a, fc1b
    except RuntimeError as re:
        raise RuntimeError(f"SVD compression failed: {re}")


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    logger: logging.Logger,
) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module,
    logger: logging.Logger,
) -> float:
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    logger.info(f"Test Loss: {test_loss:.4f} Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def validate_predictions(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    logger: logging.Logger,
) -> None:
    """
    Run the model on validation inputs and log prediction statistics.
    """
    model.eval()
    all_preds: List[torch.Tensor] = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
    all_preds = torch.cat(all_preds)
    unique, counts = torch.unique(all_preds, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"Prediction distribution (ŷ): {distribution}")


def finetune_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    finetune_epochs: int,
    logger: logging.Logger,
) -> None:
    """
    Fine-tune the compressed model on a subset of training data.
    """
    model.train()
    for epoch in range(1, finetune_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(f"Finetune Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")


def main() -> None:
    logger = setup_logger("api.log")
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the full training and test datasets.
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize and train the full network.
    full_net = FullNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(full_net.parameters(), lr=config.learning_rate)

    logger.info("Starting training of FullNet")
    for epoch in range(1, config.epochs + 1):
        train(full_net, device, train_loader, optimizer, criterion, epoch, logger)
        test(full_net, device, test_loader, criterion, logger)

    full_accuracy = test(full_net, device, test_loader, criterion, logger)
    logger.info(f"Full network accuracy: {full_accuracy * 100:.2f}%")

    k = config.svd_rank
    logger.info(f"Compressing fc1 with rank: {k}")
    fc1a, fc1b = compress_fc_layer(full_net.fc1, k)

    # logger.info(f"Compressing fc2 with rank: {k}")
    # fc2a, fc2b = compress_fc_layer(full_net.fc2, k)
    # compressed_net = CompressedNet(fc1a, fc1b, fc2a, fc2b).to(device)

    compressed_net = CompressedNet(fc1a, fc1b, full_net.fc2, full_net.fc3).to(device)

    comp_accuracy = test(compressed_net, device, test_loader, criterion, logger)
    logger.info(f"Compressed network accuracy before fine-tuning: {comp_accuracy * 100:.2f}%")

    subset_size = int(0.1 * len(train_dataset))  # use 10% of training data
    subset_indices = list(range(subset_size))
    train_subset = Subset(train_dataset, subset_indices)
    train_subset_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)

    optimizer_ft = optim.Adam(compressed_net.parameters(), lr=config.learning_rate / 10)
    logger.info("Starting fine-tuning on compressed network")
    finetune_model(compressed_net, device, train_subset_loader, optimizer_ft, criterion, config.finetune_epochs, logger)
    finetuned_accuracy = test(compressed_net, device, test_loader, criterion, logger)
    logger.info(f"Finetuned compressed network accuracy: {finetuned_accuracy * 100:.2f}%")

    validate_predictions(compressed_net, device, test_loader, logger)


if __name__ == "__main__":
    main()
