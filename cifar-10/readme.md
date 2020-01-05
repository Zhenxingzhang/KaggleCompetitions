## Problem : Object Recognition in Images

[CIFAR-10](https://www.kaggle.com/c/cifar-10/overview) is an established computer-vision dataset used for object recognition. 


## Data
It consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. 
It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

| Name | Total numbers | Image Size | Classes |
| :---: | :---: | :---: | :---: | 
|Training  | 50,000 | 32x32x3 | 10|
|Validation | 10,000 | 32x32x3 | 10|
|Test| 300,000 | 32x32x3 | 10 |


## Convolutional neural network

Classic computer vision task for image classification.


### Networks:

Convolutional layers:
- Kernel size
- Use_bias: bias or not
- Number of output filters per layer
- Initialization:
    - Kernel Weight  
    - Bias
- Regularizer:
    - kernal
    - bias
    - activity_regularizer
    
Activation layer

BatchNorm layer
- Apply BN after activation

Max-pooling layer

Dropout layer

Dense layer

### Optimizer

### Loss/Target function


## Training neural networks

- Data Augmentation
- Hyper-parameter tuning
    - Learning rate
    - Epochs
    - Batch size
    - keep_probability


## Results analysis
- Confusion matrix
- What are we expecting and does things work as we expected.

## Results

| Model-Id | Accuracy | Pre-processing | Architecture | Hyper-parameters  |
| ------|-------|--- |:-------------:| -----:|

