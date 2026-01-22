import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average="macro")
conf_matrix = confusion_matrix(y_test, y_pred)

X_vis = X.iloc[:, :2]
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(
    X_vis, y, test_size=0.2, random_state=42, stratify=y
)

model_v = LogisticRegression(max_iter=200, multi_class="ovr")
model_v.fit(X_train_v, y_train_v)

x_min, x_max = X_vis.iloc[:, 0].min() - 1, X_vis.iloc[:, 0].max() + 1
y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model_v.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z)
plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y)
plt.xlabel(X_vis.columns[0])
plt.ylabel(X_vis.columns[1])
plt.title("Decision Boundary")
plt.show()

z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure()
plt.plot(z, sigmoid)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Curve")
plt.grid()
plt.show()

plt.figure()
plt.imshow(conf_matrix)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

classes = iris.target_names
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center")

plt.show()

plt.figure()
metrics = ["Accuracy", "Recall"]
values = [accuracy, recall]

plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("Model Performance")
plt.ylabel("Score")

for i, v in enumerate(values):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

plt.show()
