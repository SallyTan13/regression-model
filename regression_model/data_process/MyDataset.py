import torch.utils.data.Dataset
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self,filepath):
        z = np.loadtxt(filepath,dtype=np.float32,delimiter = ",")
        self.x_data = torch.from_numpy(z[:,:-3])
        self.y_data = torch.from_numpy(z[:,-3:])
        self.length = z.shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self,item):
        return self.x_data[item],self.y_data[item]
    
    