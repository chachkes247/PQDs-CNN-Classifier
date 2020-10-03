# PQDs-CNN-Classifier
This repository contains a Deep CNN classifier of Power Quality Disturbances.

### Requirements:
Python 3
Tensorflow
Keras
matplotlib
scipy (using MATLAB)

### To use the code with a given database:
1) Clone the repository.
2) Add your dataset to the same folder as the cloned code. 
3) DeepCNN.py is the main file.Current dataset format is a MATLAB sturcut file. Replace the dataset filename in DeepCNN.py, at line: < DataBase = loadmat('DB_name.mat') >
