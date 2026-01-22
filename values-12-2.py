import numpy as np

m = np.array([[3,4,6],
              [2,9,7],
              [1,5,3]])

rank = np.linalg.matrix_rank(m)
det = np.linalg.det(m)
inv = np.linalg.inv(m)

print("Rank:", rank)
print("Determinant:", det)
print("Inverse:\n", inv)
