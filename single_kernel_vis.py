"""
Visualizes the first convolutional kernel of the model from exp1
Only visualize the first 16 3x3 kernel from R channel
"""

import torch
import matplotlib.pyplot as plt

from models import CIFAR10Net

# model initialization
model = CIFAR10Net()

# load the model
state_dict = torch.load("models/exp1_epochs125_seed42_model.pth")
model.load_state_dict(state_dict)
model.eval()

# extract the first 
conv_layer = model.features[0]

# get the weight
weights = conv_layer.weight.data.cpu()  
# print(weights.shape)  # size: [64, 3, 3, 3]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < 16:
        ax.imshow(weights[i, 0], cmap='gray')  # the ith 3x3 kernel from R channel
        ax.set_title(f'Kernel {i}')
        ax.axis('off')
plt.tight_layout()

plt.savefig("figures/kernel_vis.png")
plt.close()