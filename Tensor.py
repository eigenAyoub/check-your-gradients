import numpy as np
from numpy.random import randint


class Tensor(object):
    """Docstring for Tensor. """

    def __init__(self, data, label="", roots=None, how=""):
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
        output.how = "+"
        
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
        output.how = "*"
        
        def pass_the_gradient(self):
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad  = output.grad
            other.grad = output.grad

        output.pass_the_grad = pass_the_gradient
        return  output 
    
    def __matmul__(self, other):
        output = Tensor(self.data @ other.data)
        output.roots = [self, other] 
        output.how = "@"

        
        def pass_the_gradient(self):
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad  = output.grad @ other.data.T
            other.grad = self.data.T @ output.grad 
        output.pass_the_grad = pass_the_gradient
        
        return  output 

    def sigmoid(self):
        output = Tensor(1/(1+np.exp(-self.data)), label="σ("+self.label+")")
        output.roots = [self] 

        output.how = "sigmoid"
        def pass_the_gradient(self):
            """this function does:
            > Updates the gradients of the parents/roots.
            """
            self.grad = output.data*(1-output.data)
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


import graphviz


X = Tensor(randint(10, size=(5,3)), label="X")
W1 = Tensor(randint(5, size=(3, 6)), label="W1")
b1 =  Tensor(np.ones((6,)), label="b1")

Z1 = X@W1; Z1.label = "Z1"
Z12 = Z1 + b1; Z12.label = "Z12"

L1 = Z12.sigmoid()

W2 = Tensor(randint(5, size=(6,6)), label="W2")
b2 = Tensor(np.zeros(6), label="b2")

Z2 = L1@W2; Z2.label="Z2"
Z22 = Z2 + b2; Z22.label = "Z22" 

L2 = Z22.sigmoid()

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






