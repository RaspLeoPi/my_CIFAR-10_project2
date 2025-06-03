import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models import VGG_A, VGG_A_BatchNorm

def compute_loss_landscape(model, criterion, loader, direction1, direction2, theta_center, resolution=20, radius=0.1):
    """
    compute the loss landscape of a single model
    parameters
        model
        criterion
        loader: often the test loader
        direction1
        direction2
        theta_center
    """
    model.eval()
    device = next(model.parameters()).device
    loss_grid = np.zeros((resolution, resolution))
    alpha = np.linspace(-radius, radius, resolution)
    beta = np.linspace(-radius, radius, resolution)
    
    for i, a in enumerate(alpha):
        for j, b in enumerate(beta):
            perturbed_theta = theta_center + a * direction1 + b * direction2
            # load the perturbed parameters back into the model
            offset = 0
            for p in model.parameters():
                size = p.numel()
                p.data.copy_(perturbed_theta[offset:offset+size].reshape(p.shape).to(device))
                offset += size
            # compute the loss
            with torch.no_grad():
                total_loss = 0
                for inputs, targets in loader:
                    outputs = model(inputs.to(device))
                    loss = criterion(outputs, targets.to(device))
                    total_loss += loss.item()
                loss_grid[i, j] = total_loss / len(loader)
    return alpha, beta, loss_grid

def visualize_compare_landscapes(model1, model2, criterion, loader, resolution=20, radius=0.1):
    """
    visualization of the two loss landscapes
    """
    # get the center point of model parameters
    theta_center1 = torch.cat([p.flatten() for p in model1.parameters()])
    theta_center2 = torch.cat([p.flatten() for p in model2.parameters()])
    
    # generate shared directions
    direction1_model1 = torch.randn_like(theta_center1).normal_(0, 1)
    direction2_model1 = torch.randn_like(theta_center1).normal_(0, 1)
    direction1_model1, direction2_model1 = direction1_model1 / direction1_model1.norm(), direction2_model1 / direction2_model1.norm()

    direction1_model2 = torch.randn_like(theta_center2).normal_(0, 1)
    direction2_model2 = torch.randn_like(theta_center2).normal_(0, 1)
    direction1_model2, direction2_model2 = direction1_model2 / direction1_model2.norm(), direction2_model2 / direction2_model2.norm()
    
    # compute the loss landscape of the two models
    alpha, beta, loss_grid1 = compute_loss_landscape(model1, criterion, loader, direction1_model1, direction2_model1, theta_center1, resolution, radius)
    alpha, beta, loss_grid2 = compute_loss_landscape(model2, criterion, loader, direction1_model2, direction2_model2, theta_center2, resolution, radius)
    
    # subplot initialization
    fig = plt.figure(figsize=(12, 5))
    
    # model 1
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(alpha, beta)
    ax1.plot_surface(X, Y, loss_grid1, cmap='viridis', edgecolor='none')
    ax1.set_title('Model 1 (without BN)', fontsize=10)
    ax1.set_xlabel('Direction 1 (α)')
    ax1.set_ylabel('Direction 2 (β)')
    ax1.set_zlabel('Loss')
    ax1.view_init(elev=30, azim=45)  # 调整视角
    
    # model 2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, loss_grid2, cmap='plasma', edgecolor='none')
    ax2.set_title('Model 2 (with BN)', fontsize=10)
    ax2.set_xlabel('Direction 1 (α)')
    ax2.set_ylabel('Direction 2 (β)')
    ax2.set_zlabel('Loss')
    ax2.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig("figures/landscape_vis.png")
    plt.close()

def get_testloader(ratio: float=1):
    """
    Return the test loader
    The ratio controls the quantity of the loaded testset
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    subset_size = int(ratio * len(testset))
    testset_partial = torch.utils.data.Subset(testset, indices=range(subset_size))
    testloader = torch.utils.data.DataLoader(
        testset_partial, batch_size=100, shuffle=False, num_workers=4)
    
    return testloader


torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = VGG_A().to(device)
model2 = VGG_A_BatchNorm().to(device)
criterion = torch.nn.CrossEntropyLoss()

# load the models
state_dict = torch.load("models/exp7_epochs125_seed42_model.pth", map_location=device)
model1.load_state_dict(state_dict)
model1.eval()
state_dict = torch.load("models/exp8_epochs125_seed42_model.pth", map_location=device)
model2.load_state_dict(state_dict)
model2.eval()

loader = get_testloader(0.01)

visualize_compare_landscapes(model1, model2, criterion, loader)