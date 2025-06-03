"""
Util functions for training networks
"""

import torch
import torchvision
import torchvision.transforms as transforms
# import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np

def set_random_seeds(seed_value=0, device='cpu'):
    if seed_value is not None:
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        if device != 'cpu': 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            # torch.backends.cudnn.deterministic = True     # if enabled, the training speed will be slow
            # torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32  # Same seed for all workers
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloaders(ratio: float=1, seed: int=None):
    """
    ratio: control the size of the dataset being used
    range from 0 to 1
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    subset_size = int(ratio * len(trainset))
    trainset_partial = torch.utils.data.Subset(trainset, indices=range(subset_size))
    if seed is not None:    # fix the seed
        g = torch.Generator()
        g.manual_seed(seed)
        trainloader = torch.utils.data.DataLoader(
            trainset_partial, batch_size=256, shuffle=True, num_workers=4, 
            worker_init_fn=seed_worker, generator=g)
        pass
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset_partial, batch_size=256, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    subset_size = int(ratio * len(testset))
    testset_partial = torch.utils.data.Subset(testset, indices=range(subset_size))
    testloader = torch.utils.data.DataLoader(
        testset_partial, batch_size=100, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


# training and testing
def train(model, device, trainloader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}', mininterval=10)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar.set_postfix(loss=train_loss / (total / len(trainloader)), 
                                accuracy=100. * correct / total, refresh=False)

    return train_loss / len(trainloader), 100. * correct / total


def test(model, device, testloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return test_loss / len(testloader), 100. * correct / total


def exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs):
    """
    A uniform template for experiments
    input: 
        trainloader
        testloader
        timestamp
        label: the label of the experiment, passed as a string, e.g., "1"
        epochs
        **kwargs
            model
            criterion
            optimizer
            scheduler (can be None)
    The output is visible in the files
    """
    # the format of the name of the saved model and img
    seed = torch.initial_seed()
    model_format_spec = "./models/{}_epochs{}_seed{}_model.pth"
    figure_format_spec = "./figures/{}_{}_seed{}_{}.png"

    expname = f"exp{label}"
    # device initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # parameters
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    # model initialization with kwargs
    model = kwargs["model"].to(device)
    criterion = kwargs["criterion"]
    optimizer = kwargs["optimizer"]


    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, trainloader, criterion, optimizer, epoch)
        test_loss, test_acc = test(model, device, testloader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
    # store the model
    torch.save(model.state_dict(), model_format_spec.format(expname, epochs, seed))
    
    # visualize the training
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(figure_format_spec.format(expname, epochs, seed, timestamp)) 
    plt.close()

    pass

