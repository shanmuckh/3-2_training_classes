import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
logits = np.random.randn(5)

exp_vals = np.exp(logits - np.max(logits))
softmax = exp_vals / np.sum(exp_vals)

classes = [f"Class {i}" for i in range(len(logits))]

plt.bar(classes, softmax)
plt.xlabel("Classes")
plt.ylabel("Probability")
plt.title("Softmax Probability Distribution")
plt.show()

print("Logits:", logits)
print("Softmax Probabilities:", softmax)
print("Sum of Probabilities:", np.sum(softmax))
