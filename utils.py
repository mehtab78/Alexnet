import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2


def save_checkpoint(model, optimizer, epoch, filename):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {epoch}")
        return model, optimizer, epoch
    else:
        print(f"No checkpoint found at {filename}")
        return model, optimizer, 0


def plot_training_metrics(history):
    """Plot training and validation loss/accuracy"""
    # Ensure plots directory exists
    plots_dir = "/home/mehtab/AlexNet/plots"
    os.makedirs(plots_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot accuracy
    ax2.plot(history["train_acc"], label="Training Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()

    # Save plot to plots directory
    output_path = os.path.join(plots_dir, "training_metrics.png")
    plt.savefig(output_path)
    print(f"Training metrics plot saved to {output_path}")
    plt.close()


def visualize_predictions(model, test_loader, device, num_images=6):
    """
    Visualize model predictions on test images
    """
    model.eval()
    classes = test_loader.dataset.classes

    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Make predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Plot images with predictions
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(min(num_images, len(images))):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        # Denormalize
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)

        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {classes[preds[i]]}\nTrue: {classes[labels[i]]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("/home/mehtab/AlexNet/visualizations/predictions.png")
    plt.close()


def generate_cam(model, img_path, output_path, device):
    """
    Generate Class Activation Map (CAM) for an image
    """
    # Load and preprocess image
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Get the model features and output
    model.eval()

    # Save the feature maps from the last convolutional layer
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Register hook to get the output of the last convolutional layer
    model.features[-2].register_forward_hook(get_activation("features"))

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        _, pred_class = torch.max(output, 1)

    # Get the weights of the final layer
    weights = model.classifier[-1].weight.data

    # Get the activations of the last convolutional layer
    cam_feature = activation["features"].squeeze().cpu()

    # Get the weights corresponding to the predicted class
    weight_class = weights[pred_class].cpu()

    # Compute the class activation map
    batch_size, num_channels, height, width = activation["features"].shape
    cam = torch.zeros(height, width)

    for i in range(num_channels):
        cam += weight_class[0, i] * cam_feature[i]

    # Apply ReLU to the CAM
    cam = F.relu(cam)

    # Normalize the CAM
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam) if torch.max(cam) > 0 else cam

    # Convert to numpy and resize to match original image
    cam = cam.numpy()
    cam = cv2.resize(cam, (224, 224))

    # Convert original image to numpy array for blending
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = img_np * 0.5 + 0.5  # Denormalize
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    # Create heatmap from CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Combine original image and heatmap
    img_rgb = np.stack([img_np] * 3, axis=-1)  # Convert grayscale to RGB
    cam_result = heatmap * 0.4 + img_rgb * 0.6
    cam_result = np.clip(cam_result, 0, 255).astype(np.uint8)

    # Save the result
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cam_result)
    plt.title("Class Activation Map")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
