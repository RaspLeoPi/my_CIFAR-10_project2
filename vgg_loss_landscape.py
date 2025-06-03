import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
from tqdm import tqdm
import argparse
import pickle

from models import (
    VGG_A, 
    VGG_A_BatchNorm
) 


def set_random_seeds(seed_value=0, device='cpu'):
    if seed_value is not None:
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        if device != 'cpu': 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32  # Same seed for all workers
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# loading the training set
def get_trainloader(ratio: float=1, seed: int=None):
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    subset_size = int(ratio * len(trainset))
    trainset_partial = torch.utils.data.Subset(trainset, indices=range(subset_size))

    if seed is not None:    # fix the seed
        g = torch.Generator()
        g.manual_seed(seed)
        trainloader = torch.utils.data.DataLoader(
            trainset_partial, batch_size=128, shuffle=True, num_workers=4, 
            worker_init_fn=seed_worker, generator=g)
        pass
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset_partial, batch_size=128, shuffle=True, num_workers=4)
    
    return trainloader


def model_def(label, lr, device):
    """
    Function for defining a model
    """
    if label == 0:
        # define VGG without BN
        model = VGG_A().to(device)
    else:
        # define VGG with BN
        model = VGG_A_BatchNorm().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    return model, optimizer


def plotter(VGG_max, VGG_min, VGG_BN_max, VGG_BN_min):
    """
    Do the plotting with current data
    Assumption: the four arrays have the same length
    """
    X = np.arange(len(VGG_max))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.grid(color='lightgray', linestyle = '-', alpha=0.4)

    plt.fill_between(X, VGG_max, VGG_min, color="green",linewidth=2, label="Standard VGG", 
                     alpha=0.8)
    plt.fill_between(X, VGG_BN_max, VGG_BN_min, color="red",linewidth=2, label="Standard VGG + BatchNorm",
                     alpha=0.5)

    plt.title("Loss Landscape")
    plt.legend(loc="upper right")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.xlim(-300, 8000)
    plt.ylim(-0.1, 2.5)

    plt.tight_layout()
    plt.savefig("figures/landscape_vis.png")
    plt.close()    

    pass


parser = argparse.ArgumentParser()
parser.add_argument("option", help="Type 1 to get the training loss data; Type 2 to plot", 
                    choices=["1","2"])
args = parser.parse_args()

if args.option == "1":
# 2D list for recording the loss value during training
    VGG_l = []
    VGG_BN_l = []

    lr_list = [1e-3, 1e-4, 2e-3, 5e-4]
    epochs = 20

    # shared parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    trainloader = get_trainloader()

    set_random_seeds(42, device)

    for lr in lr_list:
        # define the model first
        model1, optimizer1 = model_def(0, lr, device)
        model2, optimizer2 = model_def(1, lr, device)

        model1.train()
        model2.train()

        tmploss1 = []
        tmploss2 = []

        for epoch in tqdm(range(epochs), desc=f"lr={lr}"):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer1.zero_grad()
                outputs = model1(inputs)
                loss = criterion(outputs, labels)
                tmploss1.append(loss.item())
                loss.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                outputs = model2(inputs)
                loss = criterion(outputs, labels)
                tmploss2.append(loss.item())
                loss.backward()
                optimizer2.step()
                
        VGG_l.append(tmploss1)
        VGG_BN_l.append(tmploss2)


    VGG_l = np.array(VGG_l)
    VGG_BN_l = np.array(VGG_BN_l)
    VGG_max = VGG_l.max(0)
    VGG_min = VGG_l.min(0)
    VGG_BN_max = VGG_BN_l.max(0)
    VGG_BN_min = VGG_BN_l.min(0)

    # use pickle to save data
    with open("arrays.pkl", "wb") as f:
        pickle.dump((VGG_max, VGG_min, VGG_BN_max, VGG_BN_min), f)

elif args.option == "2":
    # load the arrays
    try:
        with open("arrays.pkl", "rb") as f:
            VGG_max, VGG_min, VGG_BN_max, VGG_BN_min = pickle.load(f)
    except:
        raise Exception("Run this file with argument '1' first to store arrays.pkl")
    plotter(VGG_max, VGG_min, VGG_BN_max, VGG_BN_min)
