"""
Dataset utilities for the Lightweight CNN Disaster Detection Framework.

Expected directory layout (ImageFolder-compatible)::

    data/
    ├── train/
    │   ├── Cyclone/
    │   ├── Earthquake/
    │   ├── Flood/
    │   ├── Wildfire/
    │   └── No_Disaster/
    ├── val/
    │   └── ...
    └── test/
        └── ...

Each leaf directory contains JPEG/PNG images for that class.
"""

from pathlib import Path
from typing import Dict, List, Union

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Default image size used throughout the project
IMAGE_SIZE: int = 224

# ImageNet mean/std used for normalisation (common practice for transfer learning
# and for models trained from scratch on natural images)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Return data-augmentation transforms for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Return deterministic transforms for validation / test / inference."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),   # 256 for default 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def build_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = IMAGE_SIZE,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from an ImageFolder-structured directory.

    Args:
        data_dir: Root directory containing ``train/``, ``val/``, and ``test/``
                  subdirectories.
        batch_size: Batch size for all loaders.
        num_workers: Number of data-loading workers.
        image_size: Spatial size to resize images to.
        pin_memory: Enable pinned memory for GPU training.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"`` (only keys that
        correspond to existing subdirectories are included).
    """
    data_dir = Path(data_dir)
    transform_map = {
        "train": get_train_transforms(image_size),
        "val": get_eval_transforms(image_size),
        "test": get_eval_transforms(image_size),
    }
    loaders: Dict[str, DataLoader] = {}
    for split, tfm in transform_map.items():
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        dataset = datasets.ImageFolder(root=split_dir, transform=tfm)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
        )
    return loaders


def get_class_names(data_dir: Union[str, Path], split: str = "train") -> List[str]:
    """
    Return the sorted list of class names discovered by ImageFolder.

    Args:
        data_dir: Root data directory.
        split: Which split subdirectory to inspect (``"train"``, ``"val"``, or
               ``"test"``).

    Returns:
        List of class name strings.
    """
    dataset = datasets.ImageFolder(root=Path(data_dir) / split)
    return dataset.classes
