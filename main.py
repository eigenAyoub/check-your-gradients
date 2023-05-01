import matplotlib.pyplot as plt
from sklearn import datasets

import numpy as np
from numpy.random import normal, randint

from Tensor import Tensor
from MNISTDataset import MNISTDataset
from ToyNN import NN

import torch
import torch.functional as F


from torch.utils.data import DataLoader


torch.manual_seed(17)
np.random.seed(17)

# pytorch model
inp = 8

np.random.seed(17)
torch.manual_seed(17)
net = NN(inp*inp, 100, 10)



                                                                                
W1 =Tensor(net.lin1.weight.data.numpy())
b1 =Tensor(net.lin1.bias.data.numpy()) 

W2 =Tensor(net.lin2.weight.data.numpy())
b2 =Tensor(net.lin2.bias.data.numpy())


N = 1600
batch = 32

digits = datasets.load_digits()

g_train = MNISTDataset(digits.data[:1600], digits.target[:1600], N)
data_loader = DataLoader(g_train, batch_size =  32, shuffle=True)

g_test = MNISTDataset(digits.data[1600:], digits.target[1600:], 1765-N)
test_loader = DataLoader(g_test, batch_size =  32, shuffle=True)

for epoch in range(1):
    x_, y_ =  next(iter(data_loader))
    for i in range(1):
        x, y = Tensor(x_), Tensor(y_) 

        # backward pass:
        Z1 = x@W1; Z1.label = "Z1"
        Z12 = Z1 + b1; Z12.label = "Z12"
        L1 = Z12.ReLU()

        Z2 = L1@W2; Z2.label = "Z2"
        Z22 = Z2 + b2; Z22.label = "Z22" 

        Lsoft = L2.softmax_layer(); Lsoft.label = "softm(L2)"
        l = Lsoft.cross_entropy_loss(y); l.label = "loss"

        print("Loss at iteration ", i, "is: ", np.sum(l.data))

        l.pass_the_grad()
        Lsoft.pass_the_grad()

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


