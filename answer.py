import numpy as np
import torch 
list_a = [[[2, 3, 4, 2], [1, 3, 3]]]
# print(len(list_a))
# for ii in list_a:
#     for ye in ii:
#         print(ye)

list_b = [[2, 3], [4, 5], [3, 4], [4, 5]]

num_a = np.array(list_b, dtype=np.float32)
print(num_a)
print(type(num_a))
print(num_a.shape)
print(num_a.dtype)
# list_a = [2, 3, 4, 3, 1, 4]
# print(list_a[:-1])
list_b = [[2, 3, 4, 2], [1, 3, 3], [3, 2, 1], [5, 3, 4]]
for i in list_b:
    print(i)

list_a = [[2, 3, 4], [1, 3, 3]]
print(list_a)
print(type(list_a))
num_a = np.array([[4, 5, 6, 6], [3, 2, 4, 8]], dtype=np.float64)
print(num_a)
print(type(num_a))
ten_a = torch.tensor([[3, 4, 2, 5, 4], [4, 5, 6, 4, 2]], dtype=torch.float64)
print(ten_a)
print(type(ten_a))
print(num_a.shape)
print(ten_a.size())
