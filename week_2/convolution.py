import torch
import torch.nn as nn
from torch import mode, optim
from torch.optim import Adam

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)
        weight = torch.tensor([[[[0.5]]]])
        bias = torch.tensor([0.2])
        self.conv1.weight = nn.Parameter(weight)
        self.conv1.bias = nn.Parameter(bias)
    
    def forward(self, x):
        x = self.conv1(x)
        return x
        
x = [[[[0.1, 0.3], [0.3, 0.5]]]]  #入力行列
t = [[[[0.6, 0.9], [0.5, 0.7]]]]  #教師行列
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.float32)
print("入力は" + str(x))
print("教師データは" + str(t))
model = Model()
y_pred = model(x)
print("更新前の出力: " + str(y_pred))
print("更新前の重みとバイアスは" + str(model.state_dict()))
criterion = nn.MSELoss()    #損失関数の定義(平均二乗誤差)
optimazar = optim.SGD(model.parameters(), lr=0.01)   #確率的勾配降下法でパラメータを更新する
loss = criterion(y_pred, t)    
print("backward")
loss.backward()     #誤差逆伝搬を行う
print("損失関数は" + str(loss))
print("1重みの勾配は" + str(model.conv1.weight.grad))
optimazar.step()   #重みの更新
y_pred = model(x)
print("更新後の出力: " + str(y_pred))

print("更新後の重みとバイアスは" + str(model.state_dict()))


