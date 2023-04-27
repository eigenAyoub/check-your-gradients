import numpy as np
from numpy.random import randint, normal

import graphviz

from sklearn import datasets, svm, metrics
from torch.utils.data import Dataset, DataLoader

from MNISTDataset import MNISTDataset

class Tensor(object):
    """Docstring for Tensor. """

    def __init__(self, data, label="", roots=None, how=""):
        # what would happen if you removed None?
        self.data = data
        self.label = label
        self.roots = []
        
        # Each op/Tensor would have his own backward call.
        self.pass_the_grad = lambda : None 

        # Gradients >> Now everyone can get it...
        self.grad = np.zeros(data.shape)
        

    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())

    # Operations:

    # Magic functions:

    def __sub__(self, other):
        output = Tensor(self.data - other.data, label=self.label+"-"+other.label)
        output.roots = [self, other]
        output.how = "-"
        
        def pass_the_gradient():
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            if len(self.data.shape) == 1:
                self.grad += np.sum(output.grad, axis=0) 
            else:
                self.grad += output.grad

            if len(other.data.shape) == 1:
                other.grad += -np.sum(output.grad, axis=0)
            else:
                other.grad += -output.grad

        output.pass_the_grad = pass_the_gradient

        return  output 


    def entropy(self, other):
        ypred = self.data
        yhat = other.data

        output = -np.log(yhat(

        output = Tensor(np.sum(self.data**2), label="l")
        output.grad = np.ones(self.data.shape)
        output.roots = [self]
        output.how = "Sum"
        
        def pass_the_gradient():
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad += 2*self.data

        output.pass_the_grad = pass_the_gradient

        return  output 

    def quadratic_loss(self):
        output = Tensor(np.sum(self.data**2), label="l")
        output.grad = np.ones(self.data.shape)
        output.roots = [self]
        output.how = "Sum"
        
        def pass_the_gradient():
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad += 2*self.data

        output.pass_the_grad = pass_the_gradient

        return  output 

    def __add__(self, other):
        output = Tensor(self.data + other.data, label=self.label+"+"+other.label)
        output.roots = [self, other]
        output.how = "+"
        
        def pass_the_gradient():
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            if len(self.data.shape) == 1:
                self.grad += np.sum(output.grad, axis=0)
            else:
                self.grad += output.grad

            if len(other.data.shape) == 1:
                other.grad += np.sum(output.grad, axis=0)
            else:
                other.grad +=  output.grad

        output.pass_the_grad = pass_the_gradient

        return  output 

    def __mul__(self, other):
        output = Tensor(self.data * other.data, label=self.label+"*"+other.label)
        output.roots = [self, other] 
        output.how = "*"
        
        def pass_the_gradient():
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad  += other.grad * output.grad
            other.grad += self.grad * output.grad

        output.pass_the_grad = pass_the_gradient
        return  output 
    
    def __matmul__(self, other):
        output = Tensor(self.data @ other.data)
        output.roots = [self, other] 
        output.how = "@"

        def pass_the_gradient():
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad  += output.grad @ other.data.T
            other.grad += self.data.T @ output.grad 
        output.pass_the_grad = pass_the_gradient
        
        return  output 

    def sigmoid(self):
        output = Tensor(1/(1+np.exp(-self.data)), label="σ("+self.label+")")
        output.roots = [self] 

        output.how = "sig"
        def pass_the_gradient():
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad += output.data*(1-output.data) * output.grad
        output.pass_the_grad = pass_the_gradient

        return output 

    def comp_graph(self):
        gr = []
        def comp_gr(node):
            rs = node.roots 
            if len(rs)>0:
                l, op, rs_l = node.label, node.how, [i.label for i in rs]
                gr.append((l, op, rs_l))
                for r in rs:
                    #gr.append(comp_gr(r))
                    gr.extend(r.comp_graph())
        comp_gr(self)
        return gr

    def visualize(self):
        """
        Should be able to plot a "beautiful"  computation graph 
        of the forward pass
        """
        n_e = self.comp_graph()

l0 = lambda : (forward(X)-Y).quadratic_loss()


"""
t=0
if t :
    g = Digraph(format='png')                                                      
    g.graph_attr['rankdir']='LR'  

    for n in range(len(t)):
        g.attr('node', shape='box', style='filled', color='lightgray') 
        g.node(str(n)+t[n][1] , label = t[n][1], shape= "box") 

    for n in range(len(t)): 
        g.attr('node', color='lightblue2', shape='oval') 
        g.node(t[n][0])  
         
    for n in range(len(t)): 
        g.edge(t[n][0], str(n)+t[n][1]) 
        for k in t[n][2]: 
            g.edge(str(n)+t[n][1],k) 




## PyTorch:
import torch

tx = torch.tensor(x_, dtype=torch.float64)
ty = torch.tensor(y_, dtype=torch.float64)

tw1 = torch.tensor(w1_, requires_grad=True, dtype=torch.float64)
tw2 = torch.tensor(w2_, requires_grad=True, dtype=torch.float64)
tb1 = torch.tensor(b1_, requires_grad=True, dtype=torch.float64)
tb2 = torch.tensor(b2_, requires_grad=True, dtype=torch.float64)




py_loss = lambda : ((((tx@tw1+tb1).sigmoid()@tw2+tb2).sigmoid() - ty)**2).sum() 

py_l0 = py_loss()



alpha = 0.5
for i in range(200):
    y = ((((tx@tw1+tb1).sigmoid()@tw2+tb2).sigmoid() - ty)**2).sum()
    y.backward() 
    
    with torch.no_grad():    
        py_w1 = py_w1 - alpha*py_w1.grad
        py_w2 = py_w2 - alpha*py_w2.grad
        py_b1 = py_b1 - alpha*py_b1.grad
        py_b2 = py_b2 - alpha*py_b2.grad

    py_w1.requires_grad_(True)
    py_w2.requires_grad_(True)
    py_b1.requires_grad_(True)
    py_b2.requires_grad_(True)

    pass_(); 
    
    print(py_loss(), l0())
"""
