import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
from tqdm import tqdm
import os
import json
from model import AlexNet
from dataset import get_data_loaders


def evaluate_model(model, test_loader, criterion, device="cuda"):
    """
    Evaluate model on test dataset and return metrics
    """
    model.eval()
    all_labels = []
    all_predictions = []
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            total_samples += inputs.size(0)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate metrics
    test_loss = running_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        ["NORMAL", "PNEUMONIA"],
        save_path="/home/mehtab/AlexNet/plots/confusion_matrix.png",
    )

    # Print classification report
    report = classification_report(
        all_labels, all_predictions, target_names=["NORMAL", "PNEUMONIA"]
    )
    print("Classification Report:")
    print(report)

    results = {
        "loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "y_true": all_labels.tolist(),
        "y_pred": all_predictions.tolist(),
    }

    return results


def plot_training_history(history_file, display=False):
    """
    Plot training and validation accuracy/loss graphs with enhanced presentation

    Args:
        history_file: Path to the JSON file containing training history
        display: Whether to display the plots (default: False)
    """
    # Create output directory for plots
    plots_dir = "/home/mehtab/AlexNet/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Load training history
    try:
        with open(history_file, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: History file not found at {history_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {history_file}")
        return
    except Exception as e:
        print(f"Error loading training history from {history_file}: {e}")
        return

    # Check if required keys exist in history
    required_keys = ["train_loss", "val_loss", "train_acc", "val_acc"]
    if not all(key in history for key in required_keys):
        print(f"Error: History file is missing required keys. Required keys: {required_keys}")
        print(f"Found keys: {list(history.keys())}")
        return

    # Extract metrics
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    epochs = range(1, len(train_loss) + 1)

    # Set style for better visualization
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except:
        plt.style.use("seaborn-whitegrid")  # Fallback for older matplotlib
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot training & validation loss
    ax1.plot(
        epochs, train_loss, "b-o", linewidth=2, markersize=4, label="Training Loss"
    )
    ax1.plot(
        epochs, val_loss, "r-o", linewidth=2, markersize=4, label="Validation Loss"
    )
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Add min validation loss annotation
    min_val_loss_epoch = val_loss.index(min(val_loss)) + 1
    min_val_loss = min(val_loss)
    ax1.annotate(
        f"Min: {min_val_loss:.4f}",
        xy=(min_val_loss_epoch, min_val_loss),
        xytext=(min_val_loss_epoch, min_val_loss * 1.1),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
        fontsize=9,
    )

    # Plot training & validation accuracy
    ax2.plot(
        epochs, train_acc, "b-o", linewidth=2, markersize=4, label="Training Accuracy"
    )
    ax2.plot(
        epochs, val_acc, "r-o", linewidth=2, markersize=4, label="Validation Accuracy"
    )
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add max validation accuracy annotation
    max_val_acc_epoch = val_acc.index(max(val_acc)) + 1
    max_val_acc = max(val_acc)
    ax2.annotate(
        f"Max: {max_val_acc:.4f}",
        xy=(max_val_acc_epoch, max_val_acc),
        xytext=(max_val_acc_epoch, max_val_acc * 0.97),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
        fontsize=9,
    )

    # Add overall title and adjust layout
    plt.suptitle("AlexNet Training Performance", fontsize=16, fontweight="bold", y=1.05)
    plt.tight_layout()

    # Save the plot with high DPI for better quality
    save_path = os.path.join(plots_dir, "training_history.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plots saved to {save_path}")
        
        # Verify file was actually created
        if os.path.exists(save_path):
            print(f"Verified: File exists at {save_path}")
            print(f"File size: {os.path.getsize(save_path)} bytes")
        else:
            print("Warning: File was not created despite no error!")
    except PermissionError:
        print(f"Error: Permission denied. Cannot write to {save_path}")
    except Exception as e:
        print(f"Error saving plot to {save_path}: {e}")

    if display:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix using seaborn
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.close()


def main():
    """
    Main function to test the plotting functionality
    """
    # Path to a sample history file - replace with actual path if needed
    history_file = "/home/mehtab/AlexNet/training_history.json"
    
    # Check if history file exists
    if not os.path.exists(history_file):
        print(f"Warning: History file not found at {history_file}")
        print("Creating a sample history file for testing...")
        
        # Create a sample history file for testing
        sample_history = {
            "train_loss": [2.1, 1.8, 1.5, 1.2, 0.9, 0.7],
            "val_loss": [2.2, 1.9, 1.7, 1.4, 1.2, 1.1],
            "train_acc": [0.4, 0.5, 0.6, 0.7, 0.8, 0.85],
            "val_acc": [0.35, 0.45, 0.55, 0.65, 0.7, 0.75]
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(sample_history, f)
            print(f"Sample history file created at {history_file}")
        except Exception as e:
            print(f"Error creating sample history file: {e}")
            return
    
    # Test plotting functionality
    print(f"Testing plot_training_history with {history_file}")
    plot_training_history(history_file, display=False)
    
if __name__ == "__main__":
    main()
