import torch
from torch._C import import_ir_module
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import cv2

from network_ex import Net


net = Net()
correct = 0
total = 0

# 学習により得た重みファイル(pthファイル)
net_path = '/home/ericlab/Desktop/rikulab_rinkou/AI_rinkou/chackpoints/net.pth'
net.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))

image_path = "/home/ericlab/Desktop/rikulab_rinkou/AI_rinkou/dataset/ca/"

for i in range(10):
    image_name = "image_" + str(i) + ".jpg"
    image_root = image_path + image_name
    img = cv2.imread(image_root)
    # print(type(img))
    img_resize = cv2.resize(img, dsize=(208,28))
    input = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    plt.imshow(input, cmap='gray')
    plt.show()
    input = input[np.newaxis, :, :]
    # print(input.shape)
    data = transforms.ToTensor()(input)
    # print(type(data))

    outputs = net(data)
    # print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)

    print(i)
    print(predicted)

    # total += labels.size(0)
    # correct += (predicted == labels).sum()

# print('Accuracy %d / %d = %f' % (correct, total, correct / total))

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