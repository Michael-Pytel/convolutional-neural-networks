import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
from config import DATA_DIR

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]


def get_subset_indices(dataset, k, seed=0):
    random.seed(seed)

    class_to_indices = {}

    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)

    selected = []
    for label, indices in class_to_indices.items():
        random.shuffle(indices)
        selected.extend(indices[:k])

    return selected


def get_dataloaders(batch_size, use_augmentation=True, few_shot_k=None, seed=0):
    num_workers = min(8, os.cpu_count() or 1)
    persistent = num_workers > 0

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)
        ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std)
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        DATA_DIR + "/train", transform=train_transform
    )

    if few_shot_k is not None:
        indices = get_subset_indices(train_dataset, few_shot_k, seed)
        train_dataset = Subset(train_dataset, indices)

    val_dataset = torchvision.datasets.ImageFolder(
        DATA_DIR + "/valid", transform=transform
    )

    test_dataset = torchvision.datasets.ImageFolder(
        DATA_DIR + "/test", transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent
    )

    return train_loader, val_loader, test_loader