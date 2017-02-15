
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize
from scipy.optimize import minimize, rosen, rosen_der



#ANN class with three hidden layers
# X is no of nurons
class Ann(object):
    def __init__(self,inputs,X):
        self.inputLayerSize = inputs
        self.outputLayerSize = 1
        self.hiddenLayer1Size = X
        self.hiddenLayer2Size = X
        self.hiddenLayer3Size = X
       
        
        
        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayer1Size)
        self.w2 = np.random.randn(self.hiddenLayer1Size, self.hiddenLayer2Size)
        self.w3 = np.random.randn(self.hiddenLayer2Size, self.hiddenLayer3Size)   
        
        self.w4 = np.random.randn(self.hiddenLayer3Size, self.outputLayerSize)
  
       #activation function 
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    #iterative computing of gradient to decrease costfunction
    def costwrapper(self, params, X, Y):
        self.setparams(params)
        cost = self.cost(X, Y)
        grad = self.computegradent(X, Y)
        return cost,grad
    #call back function to return final cost value after iteration and no. of iterations
    def callback(self, params):
        self.setparams(params)
        self.j.append(self.cost(self.X,self.Y))
        self.testj.append(self.cost(self.testX,self.testY))
    #training of dataset   
    def train(self,trainX, trainY,testX,testY):
        self.X = trainX
        self.Y = trainY
        self.testX = testX
        self.testY = testY
        
        
        self.j = []
        self.testj = []
        
        params0 = self.getparams()
        option = {'maxiter': 400, 'disp': True}
        
        _res = minimize(self.costwrapper, params0,jac = True, method = 'BFGS', args = (trainX, trainY),options= option, callback = self.callback)
        self.setparams(_res.x)
        self.optimizationResults = _res
        return self.forward(self.testX)
    # derivative of activation function
    def sigmoidD(self, z):
        return  np.exp(-z)/((1+np.exp(-z))*(1+np.exp(-z)))
    
    #forward propagation  
    def forward(self, dataset):
        
        self.z1 = np.dot(dataset, self.w1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w3)
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.w4)
        yhat =  self.sigmoid(self.z4)
        
        return yhat
    #cost function
    def cost(self,X,y):
        self.lamda = 0.0001
        yhat = self.forward(X)
        y = np.array(y)
        yhat = np.array(yhat)
        
        return sum(0.5*(y-yhat)*(y-yhat))+self.lamda/2
    #cost derivatives with weights(**backword prop**)
    def costP(self, X,Y):
        self.yhat = self.forward(X)
        
        self.y = -(Y-self.yhat)
       
        self.sig = self.sigmoidD(self.z4)
       
        delta4 = np.multiply(self.y,self.sig)
        delta4 = np.nan_to_num(delta4)
        
        j4 = np.dot(self.a3.T,delta4)+self.lamda*self.w4
        delta3 = np.dot(delta4,self.w4.T)*self.sigmoidD(self.z3)
        j3 = np.dot(self.a2.T,delta3)+self.lamda*self.w3
        
        delta2 = np.dot(delta3,self.w3.T)*self.sigmoidD(self.z2)
        j2 = np.dot(self.a2.T,delta2)+self.lamda*self.w2
        delta1 = np.dot(delta2,self.w2.T)*self.sigmoidD(self.z1)
        j1 = np.dot(X.T,delta1)+self.lamda*self.w1
        
        
        
        
       
        
        return j4,j3,j2,j1
    #getting weights
    def getparams(self):
        params = np.concatenate((self.w1.ravel(),self.w2.ravel(),self.w3.ravel(),self.w4.ravel()))
        return params
    #setting weights to reduce cost function
    def setparams(self, params):
        W1_start = 0
        W1_end = self.inputLayerSize*self.hiddenLayer1Size
        
        self.w1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize,self.hiddenLayer1Size))
        W2_end = W1_end+self.hiddenLayer1Size*self.hiddenLayer2Size
        self.w2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayer1Size,self.hiddenLayer2Size))
        W3_end = W2_end+self.hiddenLayer2Size*self.hiddenLayer3Size
        self.w3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayer2Size,self.hiddenLayer3Size))
        W4_end = W3_end+self.hiddenLayer3Size*self.outputLayerSize
        self.w4 = np.reshape(params[W3_end:W4_end],(self.hiddenLayer3Size,self.outputLayerSize))
    #finding cost gradient from costP(matrix approach) 
    # not important, just for checking of gradients
    def computegradent(self,X,Y):
            j4,j3,j2,j1 = self.costP(X,Y)
            return np.concatenate((j1.ravel(),j2.ravel(),j3.ravel(),j4.ravel()))
#finding gradient from first principle
#not important,just for checking of gradients
def computenumericgradient(N,X,Y):
    paramsI = N.getparams()
    numgrad = np.zeros(paramsI.shape)
    perturb = np.zeros(paramsI.shape)
    e = 0.00001
    
    for p in range(len(paramsI)):
        perturb[p] = e
        N.setparams(paramsI + perturb)
        loss2 = N.cost(X, Y)
        N.setparams(paramsI - perturb)
        loss1 = N.cost(X, Y)
        numgrad[p] = (loss2-loss1)/(2*e)
        
        perturb[p] = 0
    N.setparams(paramsI)
    return numgrad

        
