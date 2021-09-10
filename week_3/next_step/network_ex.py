import torch.nn as nn

# ニューラルネットワークを定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(112 * 112, 4000)
        self.l2 = nn.Linear(4000, 100)
        self.l3 = nn.Linear(100, 10)
        
    def forward(self, x):
        x = x.view(-1, 112 * 112)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x