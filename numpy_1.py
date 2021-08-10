import numpy as np

l_1d = [[1, 4, 3], [1, 3, 6], [2, 4, 5]]
print(l_1d[0])
array = np.array(l_1d, dtype=np.float32)
print(array.shape)

print(type(array))
print(type(array.tolist()))

print(type(array.tolist()[0]))