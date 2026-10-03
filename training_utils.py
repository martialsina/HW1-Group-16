import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    )
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        inputs = apply_augmentations(X_train, level=level) if level is not None else X_train

        optimizer.zero_grad()
        logit = model(inputs)
        loss = F.cross_entropy(logit, y_train)
        loss.backward()
        optimizer.step()

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
    model.eval()
    with torch.no_grad():
        logits = model(X_valid)
        loss = F.cross_entropy(logits, y_valid)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_valid).float().mean()
    return loss.item(), acc.item()


def run_experiment(X_train, y_train, X_valid, y_valid, learning_rate, hidden_size=50, epochs=500, level=None, seed=0):
    """
    Trains a freshly initialized model, evaluates it on the validation set and logs the results.

    Args:
        X_train, y_train (torch.Tensor): Training data and labels.
        X_valid, y_valid (torch.Tensor): Validation data and labels.
        learning_rate (float): Learning rate for the optimizer.
        hidden_size (int): Number of neurons in the hidden layer.
        epochs (int): Number of training epochs.
        level (str or None): Augmentation level, None for no augmentation.
        seed (int): Random seed, so every run starts from the same initialization.

    Returns:
        dict: The settings of the run with its train loss, validation loss and validation accuracy.
    """
    torch.manual_seed(seed)
    model = get_model(input_size=X_train.shape[1], hidden_size=hidden_size)
    train_loss = train_model(model, X_train, y_train, learning_rate, epochs=epochs, level=level)
    valid_loss, valid_acc = valid_metrics(model, X_valid, y_valid)

    augmentation = "baseline" if level is None else level
    print(f"lr={learning_rate:<8g} hidden={hidden_size:<5} aug={augmentation:<10} | "
          f"train_loss {train_loss:.4f} | valid_loss {valid_loss:.4f} | valid_acc {valid_acc:.4f}")
    return {"learning_rate": learning_rate, "hidden_size": hidden_size, "augmentation": augmentation,
            "train_loss": train_loss, "valid_loss": valid_loss, "valid_acc": valid_acc}


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
    chunk_size=128
    image = torch.cat([transform(chunk) for chunk in image.split(chunk_size)])
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


def plot_results(df, x, log_x=False, kind="line"):
    """
    Plots train/validation loss and validation accuracy as a function of one hyperparameter.

    Args:
        df (pd.DataFrame): Results table with train_loss, valid_loss and valid_acc columns.
        x (str): Column to put on the x-axis.
        log_x (bool): Use a log scale on the x-axis.
        kind (str): "line" for numeric hyperparameters, "bar" for categories.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if kind == "bar":
        df.plot.bar(x=x, y=["train_loss", "valid_loss"], rot=0, ax=ax[0], title="Loss")
        df.plot.bar(x=x, y="valid_acc", rot=0, ax=ax[1], title="Validation accuracy", legend=False)
    else:
        df.plot(x=x, y=["train_loss", "valid_loss"], logx=log_x, logy=True, marker="o", ax=ax[0], title="Loss")
        df.plot(x=x, y="valid_acc", logx=log_x, marker="o", ax=ax[1], title="Validation accuracy", legend=False)
    fig.suptitle(f"Effect of {x.replace('_', ' ')}")
    plt.tight_layout()
    plt.show()
