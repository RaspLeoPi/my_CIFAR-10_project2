# Project 2 of DL

## Description

Train a network on CIFAR-10 and apply VGG-A model.

## TODO

- VGG
  - [x] Write the model code
  - [ ] Train one sample
  - [ ] Compare the one with BN and one without BN
  - [ ] Try to visualize the landscape
- [ ] visualization of the first convolutional layer of the model of exp1
- [x] Revise the code structure
  - Create a template for more experiments
- Complete task 1: train a network on CIFAR-10 and do multiple experiments
  - [x] Try different number of neurons/filters
    - current structure: conv1->conv2->conv3->flatten->linear->linear->linear
  - [x] Try different loss functions (with different regularization)
    - Cross entropy loss
    - Cross entropy loss with label smoothing
  - [x] Try different activations
    - ReLU
    - SiLU
  - [x] Try different optimizers
    - Adam
    - SGD
    - AdamW
  - [x] Add some schedulers
    - e.g. CosineAnnealingLR for AdamW

## Problem 1

### Models to be tested

1. exp1
   1. CIFAR10Net
   2. conv1->conv2->conv3->flatten->linear->linear->linear
   3. ReLU
   4. Cross entropy loss
   5. Adam
   6. No scheduler
   7. batchnorm enabled
   8. dropout enabled
2. exp2
   1. CIFAR10Net_smaller
   2. conv1->conv2->conv3->flatten->linear->linear
   3. ReLU
   4. Cross entropy loss
   5. Adam
   6. No scheduler
   7. batchnorm enabled
   8. dropout enabled
   - Change the number of neurons
3. exp3
   1. ResNet-18
   2. conv1->conv1->conv2->conv2->conv3->conv3->conv4->conv4->avgpool->linear
   3. ReLU
   4. Cross entropy loss
   5. SGD
   6. MultiStepLR
   - Use residual connection
4. exp4
   1. ResNet-18
   2. conv1->conv1->conv2->conv2->conv3->conv3->conv4->conv4->avgpool->linear
   3. ReLU
   4. Cross entropy loss
   5. SGD
   6. CosineAnnealingLR
   - Use a different learning rate scheduler
5. exp5 (based on exp4)
   1. ResNet-18
   2. conv1->conv1->conv2->conv2->conv3->conv3->conv4->conv4->avgpool->linear
   3. SiLU
   4. Cross entropy loss
   5. SGD
   6. CosineAnnealingLR
   - Different activations
6. exp6 (based on exp4)
   1. ResNet-18
   2. conv1->conv1->conv2->conv2->conv3->conv3->conv4->conv4->avgpool->linear
   3. ReLU
   4. Cross entropy loss with label smoothing
   5. SGD
   6. CosineAnnealingLR
   - Use a different loss function
