import torch.nn as nn
import torch


class MyLoss(nn.Module):

    def __init__(self,k1,k2,k3):
        super(MyLoss,self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
    
    def forward(self,x,y):
        return torch.mean(self.k1*torch.pow(x[0]-y[0],2) + self.k2*torch.pow(x[1]-y[1],2) + self.k3*torch.pow(x[2]-y[2],2))

def testloss():
    myloss = MyLoss(1,1,1)
    print(myloss(torch.tensor([2,2,2]),torch.tensor([2.1,2.1,2.1])))

if __name__ == "__main__":
    testloss()