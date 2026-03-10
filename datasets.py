import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import DATA_DIR
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

def get_dataloaders(batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std)
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        DATA_DIR + "/train",
        transform=transform
    )

    val_dataset = torchvision.datasets.ImageFolder(
        DATA_DIR + "/valid",
        transform=transform
    )

    test_dataset = torchvision.datasets.ImageFolder(
        DATA_DIR + "/test",
        transform=transform
    )

    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader