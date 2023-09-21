from torchvision import datasets
from torchvision import transforms

datapath = "cifar100"

cifar100_train = datasets.CIFAR100(root=datapath, train=True, download=True, transform=transforms.Compose([
    transforms.RandomCrop(36, padding=4),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
]))

cifar100_val = datasets.CIFAR100(root=datapath, train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
]))