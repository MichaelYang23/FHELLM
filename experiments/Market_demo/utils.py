# utils.py
import random
from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

KEEP = {1, 2, 3, 4}
REMAP: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3}
INV_REMAP: Dict[int, int] = {v: k for k, v in REMAP.items()}

# Use the exact same normalization everywhere.
MNIST_NORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================
# Model factory
# ==============================
def construct_mlp(num_inputs: int = 784, num_classes: int = 4, seed: int = 0) -> nn.Module:
    """
    A tiny MLP for MNIST-like data.

    NOTE:
    - Default num_classes=4 (for our 4-class pipeline).
    - If you need 10-class for other experiments, pass num_classes=10 explicitly.
    """
    set_seed(seed)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, 512, bias=False),
        nn.ReLU(),
        nn.Linear(512, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, num_classes, bias=False),
    )
    return model


def filter_mnist_indices(root: str, split: str) -> Tuple[np.ndarray, np.ndarray, datasets.MNIST]:
    """
    Load MNIST (train/test), filter to KEEP={1,2,3,4}, and return:
      filtered_indices: np.ndarray of global indices kept
      remapped_labels:  np.ndarray of labels mapped to {0,1,2,3} (same length as filtered_indices)
      dataset:          the torchvision MNIST dataset object (with transforms=MNIST_NORM)

    You will pass (filtered_indices, remapped_labels, dataset) to make_subset_loader_4class.
    """
    is_train = (split == "train")
    ds = datasets.MNIST(root=root, train=is_train, download=True, transform=MNIST_NORM)
    y = ds.targets if hasattr(ds, "targets") else ds.train_labels
    y_np = y.cpu().numpy().astype(int)

    mask = np.array([yy in KEEP for yy in y_np], dtype=bool)
    idx_kept = np.where(mask)[0]
    y_remap = np.array([REMAP[int(y_np[i])] for i in idx_kept], dtype=np.int64)
    return idx_kept, y_remap, ds


class _RemappedSubset(Dataset):
    """
    A dataset wrapper that returns (image, remapped_label).
    This ensures labels are in {0,1,2,3} while sampling by global indices.
    """
    def __init__(self, base_ds: datasets.MNIST, indices: np.ndarray, filtered_indices: np.ndarray,
                 remapped_labels_all: np.ndarray):
        self.base = base_ds
        self.idxs = indices.astype(int)
        # Build a mapping: global index (in filtered pool) -> position in that pool
        self._pos = {int(g): p for p, g in enumerate(filtered_indices)}
        self._remap_all = remapped_labels_all

    def __len__(self):
        return self.idxs.size

    def __getitem__(self, i):
        gi = int(self.idxs[i])
        x, _ = self.base[gi]             # Use image; ignore original label here
        pos = self._pos[gi]              # position inside the filtered pool
        y = int(self._remap_all[pos])    # remapped label in {0..3}
        return x, y


def make_subset_loader_4class(
    dataset: datasets.MNIST,
    filtered_indices: np.ndarray,
    remapped_labels_all: np.ndarray,
    subset_indices: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader over 'subset_indices' (global indices into 'dataset'),
    returning (x, y) where y ∈ {0,1,2,3} according to REMAP.

    Must use the (filtered_indices, remapped_labels_all) produced by filter_mnist_indices(split=...).
    """
    ds = _RemappedSubset(
        base_ds=dataset,
        indices=subset_indices,
        filtered_indices=filtered_indices,
        remapped_labels_all=remapped_labels_all,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_eval_class_mask_loader(
    dataset: datasets.MNIST,
    filtered_indices: np.ndarray,
    remapped_labels_all: np.ndarray,
    wanted_original_digits: Iterable[int],
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a loader over a subset of the filtered eval pool, keeping only items whose
    ORIGINAL label is in 'wanted_original_digits' (e.g., {3} or {4}).
    Returned y are still remapped into {0..3}. This is handy to probe per-class IF.

    Example:
      te_idx_all, te_y_remap, ds_te = filter_mnist_indices(root, split="test")
      eval3_loader = get_eval_class_mask_loader(ds_te, te_idx_all, te_y_remap, {3})
    """
    # figure out original labels for the filtered pool (from dataset.targets)
    all_orig_y = np.array([int(dataset.targets[i].item()) for i in filtered_indices], dtype=int)
    mask = np.isin(all_orig_y, list(wanted_original_digits))
    kept = filtered_indices[mask]
    return make_subset_loader_4class(
        dataset=dataset,
        filtered_indices=filtered_indices,
        remapped_labels_all=remapped_labels_all,
        subset_indices=kept,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_mnist_dataloader(
    batch_size: int = 128,
    split: str = "train",
    shuffle: bool = False,
    subsample: bool = False,
    indices: Optional[Iterable[int]] = None,
    drop_last: bool = False,
):
    """
    10-class MNIST dataloader (legacy). For our 4-class pipeline prefer:
      filter_mnist_indices(...) + make_subset_loader_4class(...)

    This function is left intact for backward compatibility with old scripts.
    """
    _tfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    is_train = split == "train"
    dataset = torchvision.datasets.MNIST(
        root="/tmp/mnist/", download=True, train=is_train, transform=_tfm
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
    batch_size: int = 128,
    split: str = "train",
    shuffle: bool = False,
    subsample: bool = False,
    indices: Optional[Iterable[int]] = None,
    drop_last: bool = False,
):
    """
    FashionMNIST 10-class (legacy). Not used in the 4-class marketplace.
    """
    _tfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    is_train = split == "train"
    dataset = torchvision.datasets.FashionMNIST(
        root="/tmp/mnist/", download=True, train=is_train, transform=_tfm
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

class IndexedDataset(Dataset):
    """Wraps a dataset to return (data, target, index)."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        return data, target, idx


# -------- Optional helpers kept from your draft (can be removed if unused) --------
def partition_data_into_datasets(train_loader: DataLoader, num_datasets: int):
    """
    Partition a dataset into num_datasets non-overlapping subsets (legacy helper).
    """
    full_dataset = train_loader.dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    partition_size = dataset_size // num_datasets

    out = {}
    for i in range(num_datasets):
        start = i * partition_size
        end = (i + 1) * partition_size if i < num_datasets - 1 else dataset_size
        subset_indices = indices[start:end]
        subset = Subset(full_dataset, subset_indices)
        out[i] = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=False)
    print(f"Successfully partitioned data into {num_datasets} datasets of ~{partition_size} samples each.")
    return out


def update_model_lora(model, optimizer, criterion, update_loader, device, epochs):
    """
    Legacy: finetune on LoRA-only params. Keep if you still use it elsewhere.
    """
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = ('lora' in name)
    for ep in range(epochs):
        tot = 0.0
        for x, y in update_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tot += loss.item()
        print(f"[LoRA adapt] epoch {ep+1}/{epochs} loss={tot/len(update_loader):.4f}")
    return model
