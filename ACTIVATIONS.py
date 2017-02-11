
# coding: utf-8

# In[2]:

#ACTIVATIONS TO BE USED IN OUR ANN MODELS - Basic Ones 


# In[4]:

import numpy as np


# In[18]:

# Create linear activations
def linear(x):
    return(x)
# Create cubic activations 
def cubic(x):
    return(x*x*x)
# Create sigmoid activations
def sigmoid(x):
    return(1/(1+np.exp(-x)))
#Create softmax activation for multiclass- Centred out the input matrix by subtracting the max value to prevent potential blowup(http://cs231n.github.io/linear-classify/)
def softmax(x):
    x = x-np.max(x)
    return(np.exp(x)/np.sum(np.exp(x)))
#Create Relu activation i.e 0 for negative values and x for positive values
def relu(x):
    return(np.maximum(x,0))
def tanh(x):
    return(np.tanh(x))
def leaky_relu(x):
    return(0.1*x+0.9*relu(x))
def softmax_prime(x):
    return(softmax(z)/(1-softmax(z)))
def sigmoid_prime(x):
    return(sigmoid(x)/(1-sigmoid(x)))
    


# In[ ]:



