import torch
import torch.nn as nn


class NN(nn.Module):                                                            
    def __init__(self, n1, n2, n3):                                             
        super().__init__()                                                      
        self.lin1 = nn.Linear(n1, n2)                                           
        self.act = nn.ReLU()                                                    
        self.lin2 = nn.Linear(n2, n3)                                           
                                                                                
    def forward(self, x):                                                       
        out = self.lin1(x)                                                      
        out = self.act(out)                                                     
        out = self.lin2(out)                                                    
                                                                                
        return out 
