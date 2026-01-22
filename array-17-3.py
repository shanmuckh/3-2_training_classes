import numpy as np

arr = np.arange(1, 13)

array_3x4 = arr.reshape(3, 4)
array_2x6 = arr.reshape(2, 6)

print("3 x 4 Array:")
print(array_3x4)

print("\n2 x 6 Array:")
print(array_2x6)
