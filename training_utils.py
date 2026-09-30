import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms


def get_model(input_size=784, hidden_size=50):
    """
    Creates a simple 2-layer fully connected neural network for image classification.

    Args:
        hidden_size (int): Number of neurons in the hidden layer.

    Returns:
        torch.nn.Sequential: A PyTorch model
    """
    # write your code here
    return model


def train_model(model, X_train, y_train, learning_rate, epochs=500, level=None):
    """
    Trains a PyTorch model using the Adam optimizer and cross-entropy loss.

    Args:
        model (torch.nn.Module): The model to train.
        X_train (torch.Tensor): Input training data of shape (N, D).
        y_train (torch.Tensor): Ground truth labels of shape (N,).
        learning_rate (float): Learning rate for the optimizer.
        epochs (int, optional): Number of training epochs. Default is 40.

    Returns:
        float: The final training loss after the last epoch.
    """
    # write your code here
    return loss.item()


def valid_metrics(model, X_valid, y_valid):
    """
    Evaluates a trained model on a validation set.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        X_valid (torch.Tensor): Validation input data of shape (N, D).
        y_valid (torch.Tensor): Ground truth labels of shape (N,).

    Returns:
        Tuple[float, float]: A tuple containing:
            - validation loss (float)
            - validation accuracy (float in [0, 1])
    """
    # write your code here
    return loss.item(), acc.item()


def apply_augmentations(image, level="mild", flip_prob=0.5):
    """
    Apply a sequence of torchvision data augmentations based on the specified
    intensity level. Images are cropped-and-padded back to their original
    28x28 size, so the output shape always matches the input shape.

    Args:
        image (torch.Tensor): Input image tensor, flattened (N, 784) or (N, 1, 28, 28)
        level (str): One of "mild", "moderate", or "aggressive"
        flip_prob (float): Probability of applying horizontal flip

    Returns:
        torch.Tensor: Augmented image, flattened to (N, 784)
    """
    image = image.reshape(-1, 1, 28, 28)
    N = image.shape[0]

    if level == "mild":
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(p=flip_prob),
        ])
    elif level == "moderate":
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=3),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    elif level == "aggressive":
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomRotation(15),
        ])
    else:
        return image.reshape(N, -1)

    image = transform(image)
    return image.reshape(N, -1)


def show_batch(images):
    N = images.shape[0]
    plt.figure(figsize=(8, 8))
    for i in range(N):
        img = images[i]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
