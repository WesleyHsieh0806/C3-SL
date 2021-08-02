# Split learning implementation
Dataset: CIFAR10
Model: AlexNet

# File Description
## train_init.py
train initialization models with different batch size
**Usage:** 
>    python train_init.py --batch ${batch size} --epoch $(nof epoch)

## train.py
train models with mask
**Usage:** 
>    python train.py --batch ${batch size} --epoch $(nof epoch) --gamma $(gamma)

## utils.py
Provide the function of circular convolution and circular correlation 

