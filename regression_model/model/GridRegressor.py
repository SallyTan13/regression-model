import torch.nn as nn
import torch

class GridRegressor(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_class):
        super(GridRegressor,self).__init__()
        self.linear_1 = nn.Linear(input_dim,hidden_dim)
        self.activate = torch.relu
        self.fc1 = nn.Linear(hidden_dim,num_class)
        self.fc2 = nn.Linear(hidden_dim,num_class)
        self.fc3 = nn.Linear(hidden_dim,num_class)
    
    def forward(self,inputs):
        y = self.linear_1(inputs)
        y = self.activate(y)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        return y1,y2,y3

def test_GridRegressor():
    mlp = GridRegressor(4,8,1)
    inputs = torch.rand(3,4)
    score = mlp(inputs)
    print(score)

if __name__ == "__main__":
    test_GridRegressor()

