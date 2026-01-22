import numpy as np

logits = np.array([2.0, 1.0, 0.1])
true_class = 0

exp_vals = np.exp(logits - np.max(logits))
softmax = exp_vals / np.sum(exp_vals)

manual_loss = -np.log(softmax[true_class])

one_hot = np.zeros_like(logits)
one_hot[true_class] = 1
library_loss = -np.sum(one_hot * np.log(softmax))

print("Logits:", logits)
print("Softmax Probabilities:", softmax)
print("True Class Index:", true_class)
print("Manual Cross Entropy Loss:", manual_loss)
print("Library Cross Entropy Loss:", library_loss)
