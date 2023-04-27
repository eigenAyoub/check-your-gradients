from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    'Characterizes a dataset for PyTorch'                                       
    def __init__(self, data, N):                                                
        'Initialization'                                                        
        self.data = data                                                        
        self.list_IDs = range(N)                                                
    def __len__(self):                                                          
        'Denotes the total number of samples'                                   
        return len(self.list_IDs)                                               
                                                                                
    def __getitem__(self, index):                                               
        'Generates one sample of data'                                          
        # Select sample                                                         
        ID = self.list_IDs[index]                                               
        X = self.data.data[10]                                                  
        y = self.data.target[10]                                                
                                                                                
        return X, y

