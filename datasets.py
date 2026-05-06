"""Dataset utilities with lazy torchvision imports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Cutout:
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


def _torchvision():
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    return dset, transforms


def _stanford_roots(path: Path):
    train_root = path / 'train' / 'images'
    valid_root = path / 'evaluate' / 'images'
    test_root = path / 'test' / 'images'
    if not train_root.exists():
        raise FileNotFoundError(f'Train split not found: {train_root}')
    if not valid_root.exists():
        raise FileNotFoundError(f'Evaluate split not found: {valid_root}')
    if not test_root.exists():
        raise FileNotFoundError(f'Test split not found: {test_root}')
    return train_root, valid_root, test_root


def _build_transforms(data: str, img_size: int, aug_lv: int):
    _, transforms = _torchvision()
    if data == 'stanford_dogs':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        base_resize = [transforms.Resize((img_size, img_size))]
    elif data == 'cifar10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        base_resize = []
    elif data == 'cifar100':
        mean = [0.50707516, 0.48654887, 0.44091784]
        std = [0.26733429, 0.25643846, 0.27615047]
        base_resize = []
    else:
        raise ValueError(data)

    train_ops = list(base_resize)
    if aug_lv >= 1:
        if data == 'stanford_dogs':
            train_ops.extend([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            train_ops.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if aug_lv >= 2:
        train_ops.append(Cutout(16))

    eval_ops = list(base_resize) + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def get_dataset(data: str, path: str, aug_lv: int, img_size: int = 128):
    dset, _ = _torchvision()
    data = data.lower().strip()
    root = Path(path)
    train_tf, eval_tf = _build_transforms(data, img_size, aug_lv)

    if data == 'cifar10':
        train_data = dset.CIFAR10(root, train=True, download=True, transform=train_tf)
        valid_data = dset.CIFAR10(root, train=False, download=True, transform=eval_tf)
        data_shape = (3, 32, 32, 10)
    elif data == 'cifar100':
        train_data = dset.CIFAR100(root, train=True, download=True, transform=train_tf)
        valid_data = dset.CIFAR100(root, train=False, download=True, transform=eval_tf)
        data_shape = (3, 32, 32, 100)
    elif data == 'stanford_dogs':
        train_root, valid_root, _ = _stanford_roots(root)
        train_data = dset.ImageFolder(train_root, transform=train_tf)
        valid_data = dset.ImageFolder(valid_root, transform=eval_tf)
        n_classes = len(train_data.classes)
        data_shape = (3, img_size, img_size, n_classes)
    else:
        raise ValueError(data)
    return train_data, valid_data, data_shape


def get_split_dataset(data: str, path: str, split: str, img_size: int = 128):
    dset, _ = _torchvision()
    data = data.lower().strip()
    split = 'evaluate' if split == 'valid' else split
    root = Path(path)
    _, eval_tf = _build_transforms(data, img_size, aug_lv=0)

    if data == 'cifar10':
        dataset = dset.CIFAR10(root, train=(split != 'test'), download=True, transform=eval_tf)
        data_shape = (3, 32, 32, 10)
    elif data == 'cifar100':
        dataset = dset.CIFAR100(root, train=(split != 'test'), download=True, transform=eval_tf)
        data_shape = (3, 32, 32, 100)
    elif data == 'stanford_dogs':
        train_root, valid_root, test_root = _stanford_roots(root)
        split_root = {'train': train_root, 'evaluate': valid_root, 'test': test_root}[split]
        dataset = dset.ImageFolder(split_root, transform=eval_tf)
        data_shape = (3, img_size, img_size, len(dataset.classes))
    else:
        raise ValueError(data)
    return dataset, data_shape
