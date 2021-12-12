import torch
import numpy as np
from torchvision import datasets, transforms


def load_cifar10_data(dir_path, train_batchsize, test_batchsize):
    # size: (32, 32, 3)
    # 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
    # train: 5000 images per class
    # test: 1000 images per class
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=dir_path + '/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batchsize, shuffle=True)

    testset = datasets.CIFAR10(root=dir_path + '/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize, shuffle=False)
    return trainloader, testloader


def load_cifar100_data(dir_path, train_batchsize, test_batchsize):
    # size: (32, 32, 3)
    # 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
    # train: 5000 images per class
    # test: 1000 images per class
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR100(root=dir_path + '/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batchsize, shuffle=True)

    testset = datasets.CIFAR100(root=dir_path + '/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize, shuffle=False)
    return trainloader, testloader


def load_mnist_data(dir_path, train_batchsize, test_batchsize):
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = datasets.MNIST(root=dir_path + '/data/minist/', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batchsize, shuffle=True)

    testset = datasets.MNIST(root=dir_path + '/data/minist/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize, shuffle=False)

    return trainloader, testloader


def load_svhn_data(dir_path, train_batchsize, test_batchsize):
    
    transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.SVHN(root=dir_path + '/data/svhn/', download=True, transform=transform, split='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batchsize, shuffle=True)

    testset = datasets.SVHN(root=dir_path + '/data/svhn/', download=True, transform=transform, split='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize, shuffle=False)

    return trainloader, testloader


def load_tiny_imagenet_data(dir_path, train_batchsize, test_batchsize, dataset=False, split_file=None):

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = dir_path + '/data/tiny_imagenet_200/train'
    val_path = dir_path + '/data/tiny_imagenet_200/test'

    if not split_file:
        split_file = dir_path + '/data/tiny_imagenet_200/npy_files/tiny-imagenet-train-val.npy'
    split_permutation = list(np.load(split_file))

    train_set = torch.utils.data.Subset(datasets.ImageFolder(train_path, transform=train_transform), split_permutation[:90000])
    val_set = torch.utils.data.Subset(datasets.ImageFolder(train_path, transform=test_transform), split_permutation[90000:])
    test_set = datasets.ImageFolder(val_path, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=train_batchsize, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batchsize, shuffle=False, num_workers=2, pin_memory=True)

    if dataset:
        print('return train dataset')
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        return train_dataset, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader


def tiny_loader(dir_path, train_batchsize, test_batchsize):
    print('start_loading!')
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
    print('start_loading_training_data!')
    trainset = datasets.ImageFolder(root=dir_path + '/train', transform=transform_train)
    print('end_loading_training_data!')
    print('start_loading_testing_data!')
    testset = datasets.ImageFolder(root=dir_path + '/val', transform=transform_test)
    print('end_loading_testing_data!')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batchsize, shuffle=False)
    
    return train_loader, test_loader




