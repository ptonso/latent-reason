import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import logging
from dataclasses import dataclass
from typing import Tuple, List

from src.logger import setup_logger
from svd_forward import *



def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    logger: logging.Logger,
    num_batches: int,
) -> float:
    """
    Benchmark inference speed over a number of batches.
    Returns the average inference time per sample in seconds.
    """
    model.eval()
    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            # For CUDA, synchronize to get accurate timing.
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(data)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)
            total_samples += data.size(0)
    avg_time = total_time / total_samples
    logger.info(f"Average inference time per sample: {avg_time * 1000:.4f} ms over {total_samples} samples")
    return avg_time


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

    # Compress the first fully connected layer using SVD.
    k = config.svd_rank
    logger.info(f"Compressing fc1 with rank: {k}")
    fc1a, fc1b = compress_fc_layer(full_net.fc1, k)

    # Create the compressed network using the SVD factors for the first layer.
    compressed_net = CompressedNet(fc1a, fc1b, full_net.fc2, full_net.fc3).to(device)
    comp_accuracy = test(compressed_net, device, test_loader, criterion, logger)
    logger.info(f"Compressed network accuracy before fine-tuning: {comp_accuracy * 100:.2f}%")

    # Fine-tune the compressed network using a subset of training data.
    subset_size = int(0.1 * len(train_dataset))  # use 10% of training data
    subset_indices = list(range(subset_size))
    train_subset = Subset(train_dataset, subset_indices)
    train_subset_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)

    optimizer_ft = optim.Adam(compressed_net.parameters(), lr=config.learning_rate / 10)
    logger.info("Starting fine-tuning on compressed network")
    finetune_model(compressed_net, device, train_subset_loader, optimizer_ft, criterion, config.finetune_epochs, logger)
    finetuned_accuracy = test(compressed_net, device, test_loader, criterion, logger)
    logger.info(f"Finetuned compressed network accuracy: {finetuned_accuracy * 100:.2f}%")

    # Validate predictions on the fine-tuned compressed network.
    validate_predictions(compressed_net, device, test_loader, logger)



    # Benchmark inference times.
    logger.info("Benchmarking inference time on full network")
    full_inference_time = benchmark_inference(full_net, device, test_loader, logger, config.benchmark_batches)
    
    logger.info("Benchmarking inference time on compressed network")
    compressed_inference_time = benchmark_inference(compressed_net, device, test_loader, logger, config.benchmark_batches)
    
    speedup = full_inference_time / compressed_inference_time if compressed_inference_time > 0 else float('inf')
    logger.info(f"Inference speedup factor: {speedup:.2f}x")





if __name__ == "__main__":
    main()
