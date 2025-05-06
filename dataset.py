import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(data_dir, batch_size=32):
    """
    Create and return train, validation, and test DataLoaders
    """
    # Data augmentation and normalization for training
    # Just normalization for validation and test
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
    }

    # Create datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val", "test"]
    }

    # Create dataloaders
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=4,
        )
        for x in ["train", "val", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    return train_loader, val_loader, test_loader, dataset_sizes
