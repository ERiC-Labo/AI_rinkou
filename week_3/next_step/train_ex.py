from os import initgroups
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network_ex import Net

# データセットをダウンロード(download=True)
# 学習データを用意
train_data = MNIST('~/Desktop/rikulab_rinkou/AI_rinkou/dataset_scaleup/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data,
                         batch_size=4,
                         shuffle=True)


net = Net() # ネットワーク定義
net = net.to("cuda")
criterion = nn.CrossEntropyLoss() # 損失関数(loss)を定義
optimizer = optim.SGD(net.parameters(), lr=0.01) # 最適化手法

save_path = "/home/ericlab/Desktop/rikulab_rinkou/AI_rinkou/chackpoints/net_ex.pth" # 重みファイルの保存

# 学習スタート
for epoch in range(20): # 何回繰り返すか(epoch数指定)
    running_loss = 0.0
    for i, data in enumerate(train_loader): # dataloaderで取り込んだデータセットを一つ一つネットワークに入力(学習)
        inputs, labels = data # 入力データと真値を格納
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        optimizer.zero_grad () # 勾配情報をリセット
        outputs = net(inputs) # 順伝播
        loss = criterion(outputs, labels) # ロスの計算
        loss.to("cuda")
        loss.backward() # 逆伝播
        optimizer.step() # パラメータの更新
        running_loss += loss.item() # lossの足し算

        # print("loss: %.3f" %(loss.item()))

        print(i)
        torch.save(net.state_dict(), save_path) # 重みファイルの保存

        # 最後にlossの出力
        if i % 1000 == 0:
            if i == 0:
                continue
            print('epoch:%d iter:%d loss:%.3f' % (epoch + 1, i, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')
