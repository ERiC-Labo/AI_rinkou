import cv2
import torch
from torchvision import transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from network_60000 import Net


image_path = "/home/ericlab/Desktop/rikulab_rinkou/AI_rinkou/dataset/camera/"
net = Net()

for i in range(10):
    image_name = "image_" + str(i) + ".jpg"
    image_root = image_path + image_name
    img = cv2.imread(image_root)
    print(type(img))
    img_resize = cv2.resize(img, dsize=(200,200))
    input = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    input = input[np.newaxis, :, :]
    print(input.shape)
    data = transforms.ToTensor()(input)
    print(type(data))

    # cv2.imshow('image',img_resize)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    outputs = net(data)
    # print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)

    print(i)
    
print(inputs[0].shape)
result = np.array(inputs[0])
result = np.squeeze(result)
print('Label:', predicted[0])
plt.imshow(result, cmap='gray')
plt.show()
    