#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.datasets import mnist
(X_train_2D, y_train_digit), (X_test_2D, y_test_digit) = mnist.load_data()
X_train = np.zeros((np.shape(X_train_2D)[0],784))
y_train = np.zeros((np.shape(y_train_digit)[0],10))
X_test = np.zeros((np.shape(X_test_2D)[0],784))
# For each training & testing sample, flatten and normalize the 28x28 inputs into 784-element arrays ranging from 0 to 1, and
# for the training sample only, create a 10 element desired output array where each element corresponds to a digit from 0 to 9
for num in range(np.shape(X_train)[0]):
    X_train[num] = X_train_2D[num].flatten()/255
    y_train[num][y_train_digit[num]] = 1
for num in range(np.shape(X_test)[0]):
    X_test[num] = X_test_2D[num].flatten()/255


# In[28]:


# Define a backpropagation neural network class
class BPNN:
    # Define the cosntructor that takes the number of hidden nodes, learning rate, momentum rate, and the initial weight range
    def __init__(self,num_hidden=30,num_epoch=5,learn_rate=0.01,momentum_rate=0.3,init_W_range=1):
        self.num_input = 784
        self.num_hidden = num_hidden
        self.num_output = 10
        self.num_epoch = num_epoch
        self.learn_rate = learn_rate
        self.momentum_rate = momentum_rate
        self.W_output = np.zeros((self.num_output,self.num_hidden+1))
        self.W_hidden = np.zeros((self.num_hidden,self.num_input+1))
        self.init_weights(init_W_range)
    # Define a function to initialize the weights using the initial weight range
    def init_weights(self,init_W_range):
        for j in range(self.num_output):
            for h in range(self.num_hidden+1):
                self.W_output[j,h] = np.random.uniform(-init_W_range,init_W_range,1)
        for h in range(self.num_hidden):
            for i in range(self.num_input+1):
                self.W_hidden[h,i] = np.random.uniform(-init_W_range,init_W_range,1)
    # Define a function to compute the hidden node outputs (used for backpropagation) and the neural network outputs for a given
    # input vector
    def compute(self,X):
        hidden_activation = np.dot(self.W_hidden[:,1:],X)+self.W_hidden[:,0] # Compute hidden node activations
        hidden_output = self.sigmoid(hidden_activation) # Compute hidden node outputs
        activation = np.dot(self.W_output[:,1:],hidden_output)+self.W_output[:,0] # Compute output node activations
        output = self.sigmoid(activation) # Compute outputs
        return hidden_output,output
        
    def sigmoid(self,A):
        Y = np.zeros(np.shape(A)[0])
        for i in range(np.shape(A)[0]):
            Y[i] = 1/(1+np.exp(-A[i]))
        return Y
        
    def train(self,X_train,y_train):
        num_samples = np.shape(X_train)[0]
        for epoch in range(self.num_epoch): # Iterate over each epoch
            # Initialize the previous weight changes to 0 for implementing momentum
            change_W_output_previous = np.zeros((self.num_output,self.num_hidden+1))
            change_W_hidden_previous = np.zeros((self.num_hidden,self.num_input+1))
            for sample in range(num_samples): # Iterate over each training sample
                X = X_train[sample] # Neural network input
                y = y_train[sample] # Desired output
                hidden_output,output = self.compute(X) # Compute hidden outputs and neural network outputs
                # Initialize the backpropagation variables for the output & hidden layers
                delta_j = np.zeros(self.num_output)
                delta_h = np.zeros(self.num_hidden)
                for j in range(self.num_output):
                    # Compute output layer backpropagation variables
                    delta_j[j] = (y[j]-output[j])*output[j]*(1-output[j])
                for h in range(self.num_hidden):
                    # Compute hidden layer backpropagation variables
                    delta_h[h] += np.dot(delta_j,self.W_output[:,h+1])*hidden_output[h]*(1-hidden_output[h])
                for j in range(self.num_output):
                    # Modify output weights and biases using backpropagation and momentum, then store the newest changes
                    change_output_bias = self.learn_rate*delta_j[j]+self.momentum_rate*change_W_output_previous[j,0]
                    self.W_output[j,0] += change_output_bias
                    change_W_output_previous[j,0] = change_output_bias
                    change_output_weight = self.learn_rate*hidden_output*delta_j[j]+self.momentum_rate*change_W_output_previous[j,1:]
                    self.W_output[j,1:] += change_output_weight
                    change_W_output_previous[j,1:] = change_output_weight
                for h in range(self.num_hidden):
                     # Modify hidden weights and biases using backpropagation and momentum, then store the newest changes
                    change_hidden_bias = self.learn_rate*delta_h[h]+self.momentum_rate*change_W_hidden_previous[h,0]
                    self.W_hidden[h,0] += change_hidden_bias
                    change_W_hidden_previous[h,0] = change_hidden_bias
                    change_hidden_weight = self.learn_rate*X*delta_h[h]+self.momentum_rate*change_W_hidden_previous[h,1:]
                    self.W_hidden[h,1:] += change_hidden_weight
                    change_W_hidden_previous[h,1:] = change_hidden_weight


# In[29]:


bpnn = BPNN() # Create a BPNN object
bpnn.train(X_train,y_train) # Train the neural network using the training data set


# In[30]:


print("TRAINING PHASE")
correct = 0
output_digit = np.zeros(np.shape(X_train)[0])
for num in range(np.shape(X_train)[0]):
    hidden_output,output = bpnn.compute(X_train[num])
    output_digit[num] = np.argmax(output) # Find training outputs
    if (output_digit[num]==y_train_digit[num]):
        correct += 1
print("Overall accuracy:")
print(100*correct/np.shape(X_train)[0],"%") # Compute overall training accuracy
# Find precision and recall statistics
print(classification_report(y_train_digit,output_digit))
# Create confusion matrix
print(confusion_matrix(y_train_digit,output_digit,labels=[0,1,2,3,4,5,6,7,8,9]))

print("")
print("TESTING PHASE")
correct = 0
output_digit = np.zeros(np.shape(X_test)[0])
for num in range(np.shape(X_test)[0]):
    hidden_output,output = bpnn.compute(X_test[num])
    output_digit[num] = np.argmax(output) # Find testing outputs
    if (output_digit[num]==y_test_digit[num]):
        correct += 1
print("Overall accuracy:")
print(100*correct/np.shape(X_test)[0],"%") # Print overall testing accuracy
# Find precision and recall statistics
print(classification_report(y_test_digit,output_digit))
# Create confusion matrix
print(confusion_matrix(y_test_digit,output_digit,labels=[0,1,2,3,4,5,6,7,8,9]))


# In[ ]:




