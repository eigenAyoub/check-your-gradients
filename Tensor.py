import numpy as np
from numpy.random import randint, normal

import graphviz


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

    # Ops:

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

        output.how = "sigmoid"
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




x = np.array([
    [5, 1, 2, 4, 0],
    [1, 4, 5, 2, 0],
    [3, 3, 3, 3, 4]])

X = Tensor(x)
Y = Tensor(np.array([[ 1],
       [0],
       [1]]))

Y.label = "Y"

W1 = Tensor(normal(scale = 1/3 ,size=(5, 10)), label="W1")
b1 =  Tensor(np.ones((10,)), label="b1")

Z1 = X@W1; Z1.label = "Z1"
Z12 = Z1 + b1; Z12.label = "Z12"

L1 = Z12.sigmoid()

W2 = Tensor(normal(scale = 1/10 , size=(10,1)), label="W2")
b2 = Tensor(np.zeros(1,), label="b2")

Z2 = L1@W2; Z2.label="Z2"
Z22 = Z2 + b2; Z22.label = "Z22" 

L2 = Z22.sigmoid()

L3 = L2-Y
loss = L3.quadratic_loss()

def forward(d):
    l2 = d@W1
    l3 = l2 + b1
    l1 = l2.sigmoid()
    return (l1@W2+b2).sigmoid()

alpha = 0.001

tt = forward(X)

for _ in range(10):
    loss.pass_the_grad()
    L3.pass_the_grad()
    L2.pass_the_grad()
    Z22.pass_the_grad()
    Z2.pass_the_grad()
    L1.pass_the_grad()
    Z12.pass_the_grad()
    Z1.pass_the_grad()

#    W1.data = W1.data - alpha*W1.grad
#    b1.data = b1.data - alpha*b1.grad

    W2.data = W2.data - alpha*W2.grad 
    b2.data = b2.data - alpha*b2.grad

#    print((forward(X)-Y).quadratic_loss())
    # Empty Grads:
    loss.grad = np.ones(loss.grad.shape)
    L3.grad = np.zeros(L3.grad.shape)

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
                          

print(tt)
print(forward(X))





llab = []
lgrad = []










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






