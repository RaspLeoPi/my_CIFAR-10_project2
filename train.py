"""
The program for training neural networks"""

import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# for redirection
import sys
import os
from datetime import datetime

# for parsing
import argparse

from models import (
    CIFAR10Net, 
    CIFAR10Net_smaller, 
    ResNet18, 
    VGG_A, 
    VGG_A_BatchNorm
)   # contains definition of various models

from utilfunc import (
    get_dataloaders, 
    exp_train, 
    set_random_seeds
)   # contains various util functions

def exp1(trainloader, testloader, timestamp, label, epochs):
    """
    Use CIFAR10Net
    CrossEntropyLoss
    Adam
    No scheduler
    """

    model = CIFAR10Net(use_bn=True, use_dropout=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4, 
    )

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)

    pass

def exp2(trainloader, testloader, timestamp, label, epochs):
    """
    Use CIFAR10Net_smaller: delete a hidden fully-connected layer
    CrossEntropyLoss
    AdamW
    No scheduler
    """

    model = CIFAR10Net_smaller(use_bn=True, use_dropout=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4, 
    )

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)
    pass

def exp3(trainloader, testloader, timestamp, label, epochs):
    """
    Use ResNet-18
    CrossEntropyLoss
    SGD
    MultiStepLR
    """

    model = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr = 0.1, 
        momentum=0.9, 
        weight_decay=5e-4, 
        nesterov=True
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)
    pass

def exp4(trainloader, testloader, timestamp, label, epochs):
    """
    Use ResNet-18
    CrossEntropyLoss
    SGD
    CosineAnnealingLR
    """
    model = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr = 0.1, 
        momentum=0.9, 
        weight_decay=5e-4, 
        nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)
    pass

def exp5(trainloader, testloader, timestamp, label, epochs):
    """
    Use ResNet-18
    CrossEntropyLoss
    SGD
    CosineAnnealingLR
    SiLU
    """
    model = ResNet18(activation=nn.SiLU)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr = 0.1, 
        momentum=0.9, 
        weight_decay=5e-4, 
        nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)
    pass

def exp6(trainloader, testloader, timestamp, label, epochs):
    """
    Use ResNet-18
    CrossEntropyLoss
    SGD
    CosineAnnealingLR
    """
    model = ResNet18()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = optim.SGD(
        model.parameters(), 
        lr = 0.1, 
        momentum=0.9, 
        weight_decay=5e-4, 
        nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)
    pass

def exp7(trainloader, testloader, timestamp, label, epochs):
    """
    VGG-A model
    """
    model = VGG_A()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4, 
    )

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)
    pass

def exp8(trainloader, testloader, timestamp, label, epochs):
    """
    VGG-A-BatchNorm
    """
    model = VGG_A_BatchNorm()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4, 
    )

    kwargs = {
        "model": model, 
        "criterion": criterion, 
        "optimizer": optimizer
    }

    # call the uniform interface
    exp_train(trainloader, testloader, timestamp, label, epochs, **kwargs)
    pass

def exp9(trainloader, testloader, timestamp, label, epochs):

    pass


def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", help="The label of the experiments")
    parser.add_argument("--no_log", help="Do not store the output as log", 
                        action="store_true")
    parser.add_argument("--epochs", help="The number of epochs", 
                        default=10, type=int)
    parser.add_argument("--seed", help="The random seed", 
                        default=42, type=int)
    args = parser.parse_args()

    # the dictionary storing the experiments
    exp_dict = {
        "1": exp1,
        "2": exp2,
        "3": exp3,
        "4": exp4,
        "5": exp5,
        "6": exp6,
        "7": exp7,
        "8": exp8,
        "9": exp9
    }

    # the log file
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    
    # dataset 
    trainloader, testloader, classes = get_dataloaders(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(args.seed, device)

    # running test
    if args.exp in exp_dict:
        verbose = f"--- Running exp{args.exp} with seed {args.seed} ---"
        print(verbose)
        if not args.no_log:
            # specify the log name of a specific experiment
            log_filename = f"exp{args.exp}_{timestamp}_seed{args.seed}_epochs{args.epochs}.txt"
            logfile = open(f"logs/{log_filename}", "w", encoding="utf-8")
            # redirection
            orig_stdout = sys.stdout
            sys.stdout = logfile
            sys.stderr = logfile
            print(f"{verbose}\n")
        # kwargs = {
        #     "epochs": args.epochs, 
        #     "seed": args.seed
        # }
        exp_dict[args.exp](trainloader, testloader, timestamp, args.exp, args.epochs)
    else:
        raise Exception("Please give correct label of experiments")

    if not args.no_log:
        logfile.close()
        sys.stdout = orig_stdout

    print(f"The training at {timestamp} is done. ")


if __name__ == '__main__':
    main()