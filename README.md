# C3-SL: Circular Convolution-Based Batch-Wise Compression for Communication-Efficient Split Learning

   This repository provides the official Pytorch implementation for **C3-SL**.
   
   [**:paperclip: Paper Link**](#citations)
   [**:pencil2: Citations**](#citations)
   
   <div align="center">
  <img width="85%" alt="C3-SL Framework" src="C3-SL_Framework.png">
   </div>
   
   
   * **Batch-Wise Compression (A new Compression Paradigm for Split Learning)**
   * **Exploit Circular Convolution and Orthogonality of features to avoid information loss**
   * **Reduce 1152x memory and 2.25x computation overhead compared to the SOTA dimension-wise compression method**
---


  <h2> Table of Contents</h2>
  <ul>
    <li>
      <a href="#books-prepare-dataset">Prepare Dataset</a>
      <ul>
        <!-- <li><a href="#built-with">Built With</a></li> -->
      </ul>
    </li>
    <li>
      <a href="#running-usage---training">Usage</a>
    </li>
    <li>
      <a href="#citations">Citations</a>
    </li>
  </ul>



---

## :books: Prepare Dataset
   Please refer to [Pretrained_Dataset](./Pretrained_Dataset.md) and [Downstream Tasks](#bicyclist-downstream-tasks) for further details.
   
   | Tasks | Datasets:point_down: |
   | - | - | 
   | Pre-Training | [ImageNet](https://image-net.org/index.php) <br> [COCO](https://cocodataset.org/#home) |
   | Downstream | [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) <br> [COCO](https://cocodataset.org/#home) |

## :running: Usage - Training
### Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) = 1.4.0
- torchvision = 0.5.0
- CUDA 10.1 (Check with nvcc --version)
- scipy, pandas, numpy
   

### Training with the [shell script]().

   
## Citations
``` bash
Coming Soon
```
