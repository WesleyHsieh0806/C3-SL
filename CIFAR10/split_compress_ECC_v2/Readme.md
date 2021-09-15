# Split learning implementation
## Usage
Read the scripts [here](./train.sh)
```bash
python train.py --batch ${batch size} --epoch ${epoch} --Lambda 0.0005 --dump_path ./log
```  
or
``` bash
./train.sh
```


Arguments | Description
- | -
batch| Batch Size
epoch| Number of training epoch
Lambda| the weight for reconstruction loss
dump_path| the path to save model pickles
restore| whether or not to restore model status

## File Description
### [utils.py](./utils.py)
Provide the function of circular convolution and circular correlation

### [train.py](./train.py)
The main training process

### [model.py](./model.py)
The ECC module and the SplitAlexNet

## Other information
* Dataset: CIFAR10
* Architecture:Alexnet
