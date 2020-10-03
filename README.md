# PQDs-CNN-Classifier
This repository contains a Deep CNN classifier of Power Quality Disturbances.

### DeepCNN.py
The main file, contains dataset uploaing.
Current dataset format is a MATLAB sturcut file.

### How to Run the code with any database :
1) Download the code to a folder in your local workspace.
2) Move your dataset to the same folder 
3) Replace the dataset file name in the code: < DataBase = loadmat('DB_name.mat') >

## Model Description 

•	The representation learning part is composed out of 3 Units, each Unit includes the following standard CNN configuration:
    o	First 1D convolution layer with ReLU activation.
    o	Second 1D convolution layer with ReLU activation (identical to the first layer).
    o	Max Pooling layer with kernel 3 and stride 1.a
    o	Batch Normalization layer 
  •	The kernel size of the convolutional and pooling layers is the same: 3.
  > The motivation to use small (minimal) filters is to be able to capture both local and overall features 
  •	The size of the stride in convolutional and pooling layers is the same: 1
  > The motivation to use a small (minimal) stride is to capture the features from the entire period of the waveform. 
  •	The number of 1D-convolutional filters in the first Unit is 32 and then is grows exponentially to 64 and 128 in Units 2,3  
  > The pretty high number of filters is complementary to their small size
  •	The pooling layers are standard - they reduce the dimensionality and highlight the differences between the layers. 
  •	The last layer has a Global Max Pooling layer instead of an ordinary Max Pooling which outputs less outputs than regular pooling therefore reduces the number of output nodes 
  •	The BN layers are the offered solution to deal with overfitting and enlarge the generalization.

•	The fully connected part is composed out of 3 dense layers and 1 batch Normalization layer:
  •	The activation function of the first 2 layers is ReLU
  •	The activation function of the output layer is SoftMax
  •	The size of the first dense layer is 256 and then is drops to 128 and 16 for the output layer
  •	The BN layer is located after the first dense layer

