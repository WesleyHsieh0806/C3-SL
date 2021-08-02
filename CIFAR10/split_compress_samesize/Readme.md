# Split learning implementation
Dataset: CIFAR10
Model: Special Design
This directory use the special architecture to make input and output size the same for each layer
# File Description
## train.py
train centralized models
**Usage:** 
>    python train.py

## utils.py
Provide the function of circular convolution and circular correlation

## train_cut.py
train models with different cut layer
cut layer=i means that the model is split at the output of the i-th conv blocks (i=1~5)
**Usage:**
>    python train_cut.py --cut ${cut layer}

## plot.py
Plot the accuracy and loss curve

