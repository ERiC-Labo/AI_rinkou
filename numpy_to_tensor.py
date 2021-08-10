import numpy as np
import torch
from torchvision import transforms 


x = np.array([[1, 2, 3], [2, 4, 1]])
print(x)
print(type(x))
print(x.dtype)
img = transforms.ToTensor()(x)
num_img = torch.from_numpy(np.array(x, dtype=np.float32))
tem_img = torch.tensor(x)
print(img)
print(num_img)
print(tem_img)

print(type(img))
print(type(num_img))

print(img.shape)
print(num_img.shape)
print(img.dtype)
img = img.unsqueeze(0)
print(img.shape)