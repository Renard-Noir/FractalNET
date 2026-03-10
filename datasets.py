""" Dataset class """
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


# Cutout code is burrowed from https://github.com/quark0/darts/blob/master/cnn/utils.py
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def get_dataset(data, path, aug_lv, img_size=128):
    # dataset class
    if data == 'cifar10':
        dset_cls = dset.CIFAR10
        data_shape = (3, 32, 32, 10)
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        train_path = path
        valid_path = path
    elif data == 'cifar100':
        dset_cls = dset.CIFAR100
        data_shape = (3, 32, 32, 100)
        MEAN = [0.50707516, 0.48654887, 0.44091784]
        STD = [0.26733429, 0.25643846, 0.27615047]
        train_path = path
        valid_path = path
    elif data == 'stanford_dogs':
        # Используем ImageFolder
        dset_cls = dset.ImageFolder
        data_shape = (3, img_size, img_size, 120)
        # Средние и std для ImageNet (можно заменить на посчитанные позже)
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        # Подпапки, созданные download_stanford_dogs.py
        train_path = Path(path) / 'train' / 'images'
        valid_path = Path(path) / 'evaluate' / 'images'
    else:
        raise ValueError(data)

    # Базовые трансформации: ресайз (для Stanford Dogs) + аугментации
    transf = []
    if data == 'stanford_dogs':
        transf.append(transforms.Resize(img_size))

    if aug_lv >= 1:
        transf += [
            transforms.RandomCrop(img_size, padding=4),  # padding можно подобрать
            transforms.RandomHorizontalFlip()
        ]

    # Финальные преобразования
    trn_transforms = transforms.Compose(transf + [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(img_size) if data == 'stanford_dogs' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    if aug_lv == 2:
        trn_transforms.transforms.append(Cutout(16))  # Cutout после нормализации? Лучше до, но оставим как в оригинале

    # Загрузка данных
    if data.startswith('cifar'):
        train_data = dset_cls(train_path, train=True, download=True, transform=trn_transforms)
        valid_data = dset_cls(valid_path, train=False, download=True, transform=val_transforms)
    else:  # stanford_dogs
        train_data = dset_cls(train_path, transform=trn_transforms)
        valid_data = dset_cls(valid_path, transform=val_transforms)
	test_data = ImageFolder(root=path / 'test' / 'images', transform=val_transforms)

    return train_data, valid_data, data_shape