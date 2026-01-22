import numpy as np

logits = np.array([2.0, 1.0, 0.1])

sigmoid = 1 / (1 + np.exp(-logits))

exp_vals = np.exp(logits)
softmax = exp_vals / np.sum(exp_vals)

print("Logits:", logits)
print("Sigmoid Output:", sigmoid)
print("Sum of Sigmoid Outputs:", np.sum(sigmoid))
print("Softmax Output:", softmax)
print("Sum of Softmax Outputs:", np.sum(softmax))
