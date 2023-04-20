# Neural Network Refactoring

## Cancer Diagnosis Using Blood Microbiome Data

Recently, it has been discovered that DNA samples of microbes in human blood can be symptoms of various types of cancer in the human body.

## Data 

We have blood sample data from 355 people with the 4 most common types of cancer: colon cancer, breast cancer, lung cancer and prostate cancer. labels.csv is a label file that shows the sample names and the disease type of each person with the corresponding sample name. The data is stored in the data.csv file. Again, each row contains the sample name of the corresponding person, and the rest is the number of DNA fragments for each type of microorganism (virus or bacteria). 1836 different microorganisms appear as features.

## Model

In this project, a multi-class classification algorithm will be implemented. (A single neural network model will predict an unknown sample for any of the 4 classes.)

Classification Algorithms

As the core method, a classifier, which is a multilayer perceptron, is implemented.

The first layer of the network is the convolutional layer with a 1x1 filter.

**Multi Layer Perceptron (MLP) Activation Functions**

- ReLU
- Sigmoid
- identity
- Tanh

## Performance Measures

Precision, Recall and F2 measurements are requested from the program as the output of its performance.
