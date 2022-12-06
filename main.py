import matplotlib.pyplot as plt
from sklearn import datasets

import numpy as np

from numpy.random import normal

# data loading
dataset = datasets.load_digits()
data = dataset.data
target = dataset.target

# Constructing 1 hot encoding of the labels (I'm not sure if this what they called)
y = np.zeros((target.shape[0],10))
y[range(target.shape[0]),target] = 1

# train / test split
X1, X2 = data[:1600], data[1600:]
Y1, Y2 = y[:1600],y[1600:]


y_test = target[1600:]

batch = 16

# shapes >> X (N,64)
#        >> Y (N,)


# Layers:

# Xavier Initia (Does any one know why it works? Poke: ML Alchemy)

W1 = normal(scale = 2/64,  size=(64,70))
W2 = normal(scale = 2/70,  size=(70,300))
W3 = normal(scale = 2/300, size=(300,10))

b1 = np.zeros((70,))
b2 = np.zeros((300,))
b3 = np.zeros((10,))


# Computing the LOSS:

def quadraticLoss(pred, labels):
    """
    Compute the quadratic loss:
    L = \sum (y_i - x_i)^2
    """

    l = (pred -  labels)**2
    return np.sum(l)



# Computing the softmax layer:
def softmax(l):
    """
    compute a vectorized softmax layer

    """
    ll = np.exp(l - l.max(axis=1, keepdims=True))
    return ll / np.sum(ll, axis=1, keepdims=True)



def forward2(X):
    Z1 = X@W1 + b1
    L1 = Z1*(Z1>0)
    Z2 = L1@W2 + b2
    L2 = Z2*(Z2>0)
    Z3 = L2@W3 + b3
    L3 = Z3*(Z3>0)
    return softmax(L3)

def forward(X):
    Z1 = X@W1 + b1
    L1 = Z1*(Z1>0)
    Z2 = L1@W2 + b2
    L2 = Z2*(Z2>0)
    Z3 = L2@W3 + b3
    return 1/(1+np.exp(-Z3))


epoch = 500
batches = 100

acc = []

for _ in range(epoch): 
    for k in range(batches):

        X = X1[k*batch: (k+1)*batch]
        yy = Y1[k*batch: (k+1)*batch]
        labs = target[k*batch:(k+1)*batch]
        
        # forward pass
        Z1 = X@W1 + b1
        L1 = Z1*(Z1>0)
        Z2 = L1@W2 + b2
        L2 = Z2*(Z2>0)
        Z3 = L2@W3 + b3
        #L3 = 1/(1+np.exp(-Z3))
        #diff = (L3-yy)**2
        #loss = np.sum(diff)

        

        L3 = Z3*(Z3>0)
        L4 = softmax(L3)

        L = -np.log(L4[range(batch),labs])

#        print(np.sum(L))


        # backward pass:
        dL4 = np.copy(yy)  

        dL4[range(batch), labs] = -1/L4[range(batch),labs]

        dL4_dL3 = np.zeros((batch, 10, 10))
        for i in range(batch):
            for j in range(10):
                dL4_dL3[i,j,:] = np.array([-L4[i, j]*L4[i, k] for k in range(10)])
                dL4_dL3[i,j,j] = L4[i,j]*(1-L4[i,j])
                
        dL3 = np.zeros((batch, 10))
        for m in range(batch):
            for n in range(10):
                dL3[m,n] = dL4[m,labs[m]] * dL4_dL3[m,n,labs[m]]

        #for j in range(10):
        #    dL3[:,j] = np.diag(L4 @ dL4_dL3[:,j,:].T) 

        #######
        #######

        # dL3 = 2*(L3-yy)
        #dZ3 = L3*(1-L3)*dL3

        dZ3 = dL3 * (Z3>0)
        dW3 = L2.T @ dZ3
        db3 = np.sum(dZ3, axis=0) 
        dL2 = dZ3 @ W3.T 

        dZ2 = dL2 * (Z2>0)

        dW2 = L1.T @ dZ2 
        db2 = np.sum(dZ2, axis=0)
        dL1 = dZ2 @ W2.T 

        dZ1 = dL1 * (Z1>0) 

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
          
        # updates:
        
        alpha = 0.0001

        W3 -= alpha * dW3
        W2 -= alpha * dW2
        W1 -= alpha * dW1 

        b1 -= alpha * db1
        b2 -= alpha * db2 
        b3 -= alpha * db3


    test = forward2(X2)
    equal= np.argmax(test, axis=1) == y_test

    print(np.sum(equal)/len(equal))




"""
#        dL3_dZ3 = 1 * (Z3>0)               
#        dZ3 = dL3 * dL3_dZ3
"""

