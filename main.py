import matplotlib.pyplot as plt
from sklearn import datasets

import numpy as np
from numpy.random import normal, randint

from Tensor import Tensor
from MNISTDataset import MNISTDataset

import torch
from torch.utils.data import DataLoader

# weights:
w1_ = normal(scale = 1/3 ,size=(64, 120))                                       
b1_ = np.ones((120,))                                                           
                                                                                
w2_ = normal(scale = 1/10 , size=(120,10))                                      
b2_ = np.zeros(10,)                                                             
                                                                                
W1 = Tensor(w1_, label="W1")                                                    
b1 =  Tensor(b1_, label="b1")                                                   
                                                                                
W2 = Tensor(w2_, label="W2")                                                    
b2 = Tensor(b2_, label="b2")                                                    
                             
N = 1797
batch_size = 20

# Loading using DataSet and DataLoader:
digits = datasets.load_digits()
g = MNISTDataset(digits.data, digits.target, N)
data_loader = DataLoader(g, batch_size=16, shuffle=True)

                                                                                
alpha = 0.01                                                                    

def forward(d):                                                                 
    l1 = (d@W1 + b1).sigmoid()                                                  
    return (l1@W2+b2).sigmoid()


for i in range(100):
    x_, y_ =  next(iter(data_loader))                                           
    x, y = Tensor(x_), Tensor(y_)                                             
    # backward pass:

    Z1 = x@W1; Z1.label = "Z1"
    Z12 = Z1 + b1; Z12.label = "Z12"
    L1 = Z12.sigmoid()
    Z2 = L1@W2; Z2.label = "Z2"
    Z22 = Z2 + b2; Z22.label = "Z22" 
    L2 = Z22.sigmoid(); L2.label = "L2"


    Lsoft = L2.softmax_layer(); Lsoft.label = "softm(L2)"

    l = Lsoft.cross_entropy_loss(y); l.label = "loss"

    print("Loss at iteration ", i, "is: ", np.sum(l.data))

    l.pass_the_grad()
    Lsoft.pass_the_grad()
    L2.pass_the_grad()
    Z22.pass_the_grad()
    Z2.pass_the_grad()
    L1.pass_the_grad()
    Z12.pass_the_grad()
    Z1.pass_the_grad()

    # update the grads:

    W1.data = W1.data - alpha*W1.grad
    b1.data = b1.data - alpha*b1.grad
    W2.data = W2.data - alpha*W2.grad 
    b2.data = b2.data - alpha*b2.grad

    # Empty Grads:

    l.grad = np.ones(l.grad.shape)
    Lsoft.grad = np.zeros(Lsoft.grad.shape)

    L2.grad =  np.zeros(L2.grad.shape)
    Z22.grad = np.zeros(Z22.grad.shape)
    Z2.grad =  np.zeros(Z2.grad.shape)
    L1.grad =  np.zeros(L1.grad.shape)
    Z12.grad = np.zeros(Z12.grad.shape)
    Z1.grad =  np.zeros(Z1.grad.shape)

    W1.grad =  np.zeros(W1.grad.shape)
    b1.grad =  np.zeros(b1.grad.shape)
    W2.grad =  np.zeros(W2.grad.shape)
    b2.grad =  np.zeros(b2.grad.shape)
