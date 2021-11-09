#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
(X_train_2D, y_train_digit), (X_test_2D, y_test_digit) = mnist.load_data()
X_train = np.zeros((np.shape(X_train_2D)[0],784))
y_train = np.zeros((np.shape(y_train_digit)[0],10))
X_test = np.zeros((np.shape(X_test_2D)[0],784))
# For each training & testing sample, flatten the 28x28 inputs into 784-element 1D arrays and create a 10 element desired 
# output 1D array where each element corresponds to a digit from 0 to 9
for num in range(np.shape(X_train)[0]):
    X_train[num] = X_train_2D[num].flatten()/255
    y_train[num][y_train_digit[num]] = 1
for num in range(np.shape(X_test)[0]):
    X_test[num] = X_test_2D[num].flatten()/255


# In[23]:


# Create a sequential neural network with 2 dense layers (hidden and output layers, where "units" defines the number of nodes,
# the sigmoid activation function is used and a 784 element 1D array is the input to the first (hidden) layer
BPNN = Sequential([
    Dense(units=30,input_shape=(784,),activation='sigmoid',name='hidden_layer'),
    Dense(units=10,activation='sigmoid',name='output_layer')
])
# Define the optimizers to be gradient descent with momentum
opt = keras.optimizers.SGD(learning_rate=0.01,momentum=0.3)
# Compile the neural network
BPNN.compile(optimizer=opt,loss='mse',metrics=['accuracy'])
# Train the neural network using the training data
BPNN.fit(X_train, y_train, batch_size=1,epochs=5)


# In[24]:


# Predict outputs of the training data
print("TRAINING PHASE")
train_predictions = BPNN.predict(X_train)
train_predictions_digit = np.argmax(train_predictions, axis=1)
# Create statistics to evaluate performance
correct = 0
output_digit = np.zeros(np.shape(X_train)[0])
for num in range(np.shape(X_train)[0]):
    output_digit[num] = np.argmax(train_predictions[num])
    if (output_digit[num]==y_train_digit[num]):
        correct += 1
print("Overall accuracy:")
print(100*correct/np.shape(X_train)[0],"%")
print(classification_report(y_train_digit,output_digit))
print(confusion_matrix(y_train_digit,output_digit,labels=[0,1,2,3,4,5,6,7,8,9]))

# Predict outputs of the test data
print("")
print("TESTING PHASE")
test_predictions = BPNN.predict(X_test)
test_predictions_digit = np.argmax(test_predictions, axis=1)
# Create statistics to evaluate performance
correct = 0
output_digit = np.zeros(np.shape(X_test)[0])
for num in range(np.shape(X_test)[0]):
    output_digit[num] = np.argmax(test_predictions[num])
    if (output_digit[num]==y_test_digit[num]):
        correct += 1
print("Overall accuracy:")
print(100*correct/np.shape(X_test)[0],"%")
print(classification_report(y_test_digit,output_digit))
print(confusion_matrix(y_test_digit,output_digit,labels=[0,1,2,3,4,5,6,7,8,9]))


# In[ ]:




