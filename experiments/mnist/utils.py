import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset  # <<< --- THIS IS THE ONLY LINE TO ADD/CHANGE
from torchvision import datasets, transforms

from torch.utils.data import Dataset # Add this import

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def construct_mlp(num_inputs=784, num_classes=10, seed=0):
    set_seed(seed)
    model = torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, 512, bias=False),
        nn.ReLU(),
        nn.Linear(512, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, num_classes, bias=False),
    )
    return model


def get_mnist_dataloader(
    batch_size=128,
    split="train",
    shuffle=False,
    subsample=False,
    indices=None,
    drop_last=False,
):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    is_train = split == "train"
    dataset = torchvision.datasets.MNIST(
        root="/tmp/mnist/", download=True, train=is_train, transform=transforms
    )

    if subsample and split == "train" and indices is None:
        dataset = torch.utils.data.Subset(dataset, np.arange(6_000))

    if indices is not None:
        if subsample and split == "train":
            print("Overriding `subsample` argument as `indices` was provided.")
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=0,
        drop_last=drop_last,
    )


def get_fmnist_dataloader(
    batch_size=128,
    split="train",
    shuffle=False,
    subsample=False,
    indices=None,
    drop_last=False,
):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    is_train = split == "train"
    dataset = torchvision.datasets.FashionMNIST(
        root="/tmp/mnist/", download=True, train=is_train, transform=transforms
    )

    if subsample and split == "train" and indices is None:
        dataset = torch.utils.data.Subset(dataset, np.arange(6_000))

    if indices is not None:
        if subsample and split == "train":
            print("Overriding `subsample` argument as `indices` was provided.")
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=0,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # Verifying if datasets look reasonable.
    import matplotlib.pyplot as plt

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    loader = get_mnist_dataloader(batch_size=16, shuffle=True, subsample=True)
    data_iter = iter(loader)
    data = next(data_iter)
    imshow(torchvision.utils.make_grid(data[0], padding=2))

    loader = get_fmnist_dataloader(batch_size=16, shuffle=True, subsample=True)
    data_iter = iter(loader)
    data = next(data_iter)
    imshow(torchvision.utils.make_grid(data[0], padding=2))


def partition_data_into_datasets(train_loader, num_datasets):
    """
    Partitions the full training dataset into a specified number of smaller,
    non-overlapping datasets (DataLoaders).

    Args:
        train_loader (DataLoader): The DataLoader for the full training set.
        num_datasets (int): The number of smaller datasets to create.

    Returns:
        dict: A dictionary where keys are dataset indices (0, 1, 2...) and
              values are the corresponding DataLoader objects.
    """
    full_dataset = train_loader.dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    # Shuffle indices to ensure random distribution of samples across datasets
    np.random.shuffle(indices)
    
    # Calculate the size of each new dataset
    partition_size = dataset_size // num_datasets
    
    partitioned_datasets = {}
    for i in range(num_datasets):
        start_idx = i * partition_size
        # The last partition takes all remaining indices
        end_idx = (i + 1) * partition_size if i < num_datasets - 1 else dataset_size
        
        subset_indices = indices[start_idx:end_idx]
        subset = Subset(full_dataset, subset_indices)
        
        # Create a new DataLoader for this subset
        partitioned_datasets[i] = DataLoader(
            subset,
            batch_size=train_loader.batch_size,
            shuffle=False  # No need to shuffle within the partition
        )
        
    print(f"Successfully partitioned data into {num_datasets} datasets of ~{partition_size} samples each.")
    return partitioned_datasets


def update_model_lora(model, optimizer, criterion, update_loader, device, epochs):
    """
    Performs a few epochs of fine-tuning on the model's LoRA adapters using a
    specific dataset. This is the 'adaptation' step.

    Args:
        model (torch.nn.Module): The current MLP model with LoRA adapters.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        update_loader (DataLoader): The DataLoader for the dataset to fine-tune on.
        device (torch.device): The device to run on (CPU or CUDA).
        epochs (int): The number of epochs to fine-tune for (should be small).

    Returns:
        torch.nn.Module: The fine-tuned model.
    """
    model.train() # Set the model to training mode

    # Ensure only LoRA parameters are trainable
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    print(f"Starting adaptation: Fine-tuning on new dataset for {epochs} epoch(s)...")
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in update_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"  Adaptation Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(update_loader):.4f}")

    return model


class IndexedDataset(Dataset):
    """Wraps a dataset to return (data, target, index)."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        return data, target, idx