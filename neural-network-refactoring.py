import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Reading "data" and "labels" files:

data = pd.read_csv("data.csv")
labels = pd.read_csv("labels.csv")

# Rename the empty column to represent sample values: 
data.columns.values[0] = 'Sample'

# Separating the 'Sample' column found in the labels.csv file:
X = data.drop('Sample', axis = 1) 

# Identifying disease types as targets:
Y = labels['disease_type'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42) 
Y_train = Y_train.values.ravel() 
Y_test = Y_test.values.ravel()

# Model Shapes

print("Model Shapes:")
print("-------------")

print('Train Features Shape:', X_train.shape)
print('Train Labels Shape:', Y_train.shape)
print('Test Features Shape:', X_test.shape)
print('Test Labels Shape:', Y_test.shape)


# Scaling and equalizing data with the standardization process:
sc = StandardScaler()

# Model training:
sc.fit(X_train)

# Applying the transformation to the training data:
X_train = sc.transform(X_train)

# Applying the transformation to the test data:
X_test = sc.transform(X_test)

# 4 different outputs by changing the parameters of the MLP classifier. (mlp_1, mlp_2, mlp_3, and mlp_4)

# MLP_1 Activation Function: ReLu Function

mlp_1 = MLPClassifier(activation = 'relu', hidden_layer_sizes = (30, 30, 30), max_iter = 100, random_state = None, solver = 'lbfgs')

# Model training:
mlp_1.fit(X_train, Y_train)

# Prediction:
predictions = mlp_1.predict(X_test)


# Precision, Recall and F2 Calculation (MLP_1):

print("Performance Results (MLP_1)")
print("----------------------------")

precision = precision_score(Y_test, predictions, average = None)

print("Precision:", precision)
print('\n')

recall = recall_score(Y_test, predictions, average = None)

print("Recall:", recall)
print('\n')

F2 = (5 * precision * recall) / (4 * precision + recall)

print("F2 Results:", F2)
print("----------------------------")
print('\n')


# MLP_2 Activation Function: Logistic Sigmoid Function

mlp_2 = MLPClassifier(activation = 'logistic', hidden_layer_sizes = (150, 100, 50), max_iter = 200, random_state = 1, solver = 'adam')

mlp_2.fit(X_train, Y_train)

predictions = mlp_2.predict(X_test)


# Precision, Recall and F2 Calculation (MLP_2)

print('\n')
print("Performance Results (MLP_2)")
print("----------------------------")
precision = precision_score(Y_test, predictions, average = None)

print("Precision:", precision)
print('\n')

recall = recall_score(Y_test, predictions, average = None)

print("Recall:", recall)
print('\n')

F2 = (5 * precision * recall) / (4 * precision + recall)

print("F2 Results:", F2)
print("----------------------------")
print('\n')

# MLP_3 Activation Function: Identity Function

mlp_3 = MLPClassifier(activation = 'identity', hidden_layer_sizes = (10, 30, 20), max_iter = 300, random_state = 2, solver = 'sgd')

mlp_3.fit(X_train, Y_train)

predictions = mlp_3.predict(X_test)


# Precision, Recall and F2 Calculation (MLP_3)
print('\n')
print("Performance Results (MLP_3)")
print("----------------------------")

precision = precision_score(Y_test, predictions, average = None)

print("Precision:", precision)
print('\n')

recall = recall_score(Y_test, predictions, average = None)

print("Recall:", recall)
print('\n')

F2 = (5 * precision * recall) / (4 * precision + recall)

print("F2 Results:", F2)
print("----------------------------")
print('\n')

# MLP_4 Activation Function: Tanh Function

mlp_4 = MLPClassifier(activation = 'tanh', hidden_layer_sizes = (50, 50, 50), max_iter = 400, random_state = 3, solver = 'lbfgs')

mlp_4.fit(X_train, Y_train)

predictions = mlp_4.predict(X_test)


# Precision, Recall and F2 Calculation (MLP_4)
print('\n')
print("Performance Results (MLP_4)")
print("----------------------------")

precision = precision_score(Y_test, predictions, average = None)

print("Precision:", precision)
print('\n')

recall = recall_score(Y_test, predictions, average = None)

print("Recall:", recall)
print('\n')

F2 = (5 * precision * recall) / (4 * precision + recall)

print("F2 Results:", F2)
print("----------------------------")