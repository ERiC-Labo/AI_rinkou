import torch.nn as nn

# ニューラルネットワークを定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 50)
        self.l2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.l1(x)
        x = self.l2(x)
        return x