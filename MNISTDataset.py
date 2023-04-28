import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    'Characterizes a dataset for PyTorch'                                       
    def __init__(self, X, Y, N):                                                
        'Initialization'                                                        
        self.data = X 
        self.target = Y 
        self.list_IDs = range(N)                                                
    def __len__(self):                                                          
        'Denotes the total number of samples'                                   
        return len(self.list_IDs)                                               
                                                                                
    def __getitem__(self, index):                                               
        'Generates one sample of data'                                          
        x = self.data[index]                                                  
        y = torch.zeros(10)
        y[self.target[index]] = 1

        return x, y

