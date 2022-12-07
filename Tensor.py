import numpy as np
from numpy.random import randint


class Tensor(object):
    """Docstring for Tensor. """

    def __init__(self, data, label="", roots=None, how=None):
        # what would happen if you removed None?
        self.data = np.array(data)
        self.label = label
        self.roots = []
        
        # Each op/Tensor would have his own backward call.
        self.pass_the_grad = lambda : None 

        # Gradients >> Now everyone can get it...
        self.grad = np.zeros(self.data.shape)
        

    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())

    # Ops:
    def __add__(self, other):
        output = Tensor(self.data + other.data, label=self.label+"+"+other.label)
        output.roots = [self, other] 
        
        def pass_the_gradient(self):
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            if len(self.data.shape) == 1:
                self.grad = np.sum(output.grad, axis=0)
            else:
                self.grad = output.grad

            if len(self.data.shape) == 1:
                other.grad= np.sum(output.grad, axis=0)
            else:
                other.grad =  output.grad

        output.pass_the_grad = pass_the_gradient

        return  output 

    def __mul__(self, other):
        output = Tensor(self.data * other.data, label=self.label+"*"+other.label)
        output.roots = [self, other] 
        
        def pass_the_gradient(self):
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad  = output.grad
            other.grad = output.grad

        output.pass_the_grad = pass_the_gradient
        return  output 
    
    def __matmul__(self, other):
        output = Tensor(self.data @ other.data, label=self.label+"@"+other.label)
        output.roots = [self, other] 


        def pass_the_gradient(self):
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad  = output.grad @ other.data.T
            other.grad = self.data.T @ output.grad 
        output.pass_the_grad = pass_the_gradient
        
        return  output 

    def sigmoid(self):
        output = Tensor(1/(1+np.exp(-self.data)), label="sigma("+self.label+")")
        output.roots = [self] 

        def pass_the_gradient(self):
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad = output.data*(1-output.data)
        output.pass_the_grad = pass_the_gradient

        return output 

X = Tensor(randint(10, size=(5,3)), label="X")
W1 = Tensor(randint(5, size=(3, 6)), label="W1")
b1 =  Tensor(np.ones((6,)), label="b1")

Z1 = X@W1
Z2 = Z1 + b1

L = Z2.sigmoid()

print(f"X\n {X}")
print(f"W1\n {W1}")
print(f"b1\n {b1}")
print(f"Z1\n {Z1}")
print(f"Z2\n {Z2}")
print(f"L\n {L}")

l = [L, Z2, Z1, W1, b1]

print(type(L))
for i in l:
    print(f"Roots of {i.label}")
    print(f"{[r.label  for r in i.roots ]}")







