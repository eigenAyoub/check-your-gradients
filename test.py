import matplotlib.pyplot as plt
from sklearn import datasets

import numpy as np
from numpy.random import normal, randint

import torch


# data loading
dataset = datasets.load_digits()
data = dataset.data
target = dataset.target

# Constructing 1 hot encoding of the labels (I'm not sure if this what they ca      lled)
y = np.zeros((target.shape[0],10))
y[range(target.shape[0]),target] = 1

# train / test split
X1, X2 = data[:1600], data[1600:]
Y1, Y2 = y[:1600],y[1600:]


y_test = target[1600:]

batch = 16

print("we working baby")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class MyNN(nn.Module):
    """A basic NN"""
    def __init__(self, n1, n2, n3):
        super().__init__()
        self.l1 = nn.Linear(n1, n2)
        self.l2 = nn.Tanh()
        self.l3 = nn.Linear(n2, n3)
    
    def forward(self, x):
        """TODO: Docstring for forward.

        :f: TODO
        :returns: TODO

        """
        x = self.l1(x) 
        x = self.l2(x) 
        x = self.l3(x) 
        return x

    


