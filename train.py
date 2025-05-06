import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet
from dataset import get_data_loaders
from utils import plot_training_metrics, save_checkpoint
from tqdm import tqdm
import copy
import json

# Import the plot_training_history function from evaluate.py
from evaluate import plot_training_history


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, save_dir
):
    """
    Train the model and save checkpoints
    """
    device = next(model.parameters()).device
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=f"{phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Record history
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())
                # Step the scheduler on validation loss
                scheduler.step(epoch_loss)

                # Only save the checkpoint if it's the best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save best model
                    best_model_path = os.path.join(save_dir, "best_model.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_model_wts,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": epoch_loss,
                            "acc": best_acc,
                        },
                        best_model_path,
                    )
                    print(f"New best model saved at epoch {epoch + 1}")

                # Save final model only at the last epoch
                if epoch == epochs - 1:
                    final_model_path = os.path.join(save_dir, "final_model.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": epoch_loss,
                            "acc": epoch_acc,
                        },
                        final_model_path,
                    )
                    print(f"Final model saved at epoch {epoch + 1}")

        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create directories for checkpoints
    os.makedirs("/home/mehtab/AlexNet/checkpoints", exist_ok=True)

    # Data loaders
    data_dir = "/home/mehtab/AlexNet/chest_xray"  # Adjust this path if needed
    train_loader, val_loader, test_loader, dataset_sizes = get_data_loaders(
        data_dir, batch_size=16, val_split=0.2, test_split=0.1
    )

    print(
        f"Dataset sizes - Train: {dataset_sizes[0]}, Val: {dataset_sizes[1]}, Test: {dataset_sizes[2]}"
    )

    # Initialize model
    model = AlexNet(num_classes=2).to(device)

    # Loss function with class weighting to handle imbalance if necessary
    criterion = nn.CrossEntropyLoss()

    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Scheduler to reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    # Train the model
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=30,
        save_dir="/home/mehtab/AlexNet/checkpoints",
    )

    # Save training history to JSON file for later visualization
    history_file = os.path.join("/home/mehtab/AlexNet/checkpoints", "history.json")
    with open(history_file, "w") as f:
        json.dump(history, f)
    print(f"Training history saved to {history_file}")

    # Generate and save training plots
    plot_training_history(history_file)

    print("Training finished, model and training history saved.")


if __name__ == "__main__":
    main()
