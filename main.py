import os
import argparse
import torch
from model import AlexNet
from train import train_model
from evaluate import evaluate_model
from dataset import get_data_loaders
from utils import visualize_predictions, generate_cam
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json


def main():
    parser = argparse.ArgumentParser(description="AlexNet for Pneumonia Detection")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/mehtab/AlexNet/chest_xray",
        help="path to the chest_xray dataset",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay for L2 regularization",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "visualize"],
        help="train, evaluate or visualize the model",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="path to model checkpoint"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/mehtab/AlexNet/checkpoints",
        help="directory to save model checkpoints",
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("/home/mehtab/AlexNet/visualizations", exist_ok=True)
    os.makedirs(
        "/home/mehtab/AlexNet/plots", exist_ok=True
    )  # Ensure plots directory exists

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Get data loaders
    loaders_result = get_data_loaders(args.data_dir, batch_size=args.batch_size)

    # Handle both possible return value formats
    if len(loaders_result) == 3:
        train_loader, val_loader, test_loader = loaders_result
        # If needed, we can calculate dataset sizes here
        dataset_sizes = {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }
    else:
        train_loader, val_loader, test_loader, dataset_sizes = loaders_result

    # Initialize model
    model = AlexNet(num_classes=2).to(device)

    if args.mode == "train":
        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Scheduler to reduce LR when validation loss plateaus
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, verbose=True
        )

        # Train the model
        print("Starting training...")
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            epochs=args.epochs,  # Changed from num_epochs to epochs to match function definition
            save_dir=args.save_dir,
        )

        # Save the final model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(args.save_dir, "final_model.pth"),
        )

        # Save training history to JSON file
        history_file = os.path.join(args.save_dir, "history.json")
        with open(history_file, "w") as f:
            json.dump(history, f)
        print(f"Training history saved to {history_file}")

        # Plot training history
        from evaluate import plot_training_history

        plot_training_history(history_file)

        print("Training completed!")

    elif args.mode == "evaluate":
        # Load model checkpoint if provided
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {args.checkpoint}")
        else:
            print("No checkpoint provided, using randomly initialized model")

        # Evaluate the model
        criterion = nn.CrossEntropyLoss()
        results = evaluate_model(model, test_loader, criterion, device)

        # Create plots directory if it doesn't exist
        os.makedirs("/home/mehtab/AlexNet/plots", exist_ok=True)

        print("\nEvaluation Results:")
        print(f"Test Loss: {results['loss']:.4f}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"Test F1-score: {results['f1']:.4f}")

        # Plot training history if available
        history_file = os.path.join(args.save_dir, "history.json")
        if os.path.exists(history_file):
            from evaluate import plot_training_history

            plot_training_history(history_file)

    elif args.mode == "visualize":
        # Load model checkpoint if provided
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {args.checkpoint}")
        else:
            print("No checkpoint provided, using randomly initialized model")

        # Visualize some predictions
        visualize_predictions(model, test_loader, device)

        # Generate CAMs for a few samples
        sample_dir = os.path.join(args.data_dir, "test")
        for label in ["NORMAL", "PNEUMONIA"]:
            img_dir = os.path.join(sample_dir, label)
            for i, img_file in enumerate(
                os.listdir(img_dir)[:3]
            ):  # Process 3 images per class
                img_path = os.path.join(img_dir, img_file)
                out_path = f"/home/mehtab/AlexNet/visualizations/CAM_{label}_{i}.png"
                generate_cam(model, img_path, out_path, device)
                print(f"Generated CAM for {img_path}")


if __name__ == "__main__":
    main()
