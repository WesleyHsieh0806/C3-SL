# Split learning implementation
Dataset: CIFAR10
Model: AlexNet
Train models with circular conv/correlation reconstruction loss

# File Description
### train.py
train models with different batch size

**Usage:** 

```bash
python train.py --batch ${batch size} --epoch ${epoch} --Lambda 0.0005 --dump_path ./log
```  
or
``` bash
./train.sh
```

### utils.py
Provide the function of circular convolution and circular correlation

### plot.py
Plot the accuracy and loss curve

