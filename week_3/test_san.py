import torch
from torch._C import import_ir_module
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

from network import Net


test_data = MNIST('~/Desktop/ishiyama/AI_rinkou/dataset', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data,
                         batch_size=4,
                         shuffle=False)

# softmax = torch.nn.Softmax(1)
net = Net()
correct = 0
total = 0
i = 0

# 学習により得た重みファイル(pthファイル)
net_path = '/home/ericlab/Desktop/rikulab_rinkou/AI_rinkou/chackpoints/net.pth'
net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))

for data in test_loader:
    inputs, labels = data
    outputs = net(inputs)
    # print(outputs)
    # outputs = softmax(outputs)
    # print(outputs.shape)
    # print(torch.sum(outputs))
    _, predicted = torch.max(outputs.data, 1) 
    
    total += labels.size(0)
    correct += (predicted == labels).sum()

    i += 1
    
print('Accuracy %d / %d = %f' % (correct, total, correct / total))

test_iter = iter(test_loader)
inputs, labels = test_iter.next()
outputs = net(inputs)
_, predicted = torch.max(outputs.data, 1)

print(inputs[0].shape)
result = np.array(inputs[0])
result = np.squeeze(result)
print('Label:', predicted[0])
plt.imshow(result, cmap='gray')
plt.show()