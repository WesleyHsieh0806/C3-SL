# C3-SL on CIFAR10
## Environment
* Python 3.6
* Pytorch 1.4.0
* torchvision
* CUDA 10.2
* tensorflow_datasets
* Other dependencies: numpy, matplotlib, tensorflow

## Dataset Preparation

Use the scripts in [data preprocess src](data%20preprocess%20src/download_data.py)
```bash
$ cd "data preprocess src"    
$ python download_data.py
```
Then, the data folder will be structured like
```bash
CIFAR10(Current dir)
├──CIFAR
│   ├── train
│   ├── val
├──data preprocess src
│   ├── download_data.py
```
