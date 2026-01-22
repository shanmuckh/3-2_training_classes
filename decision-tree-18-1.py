import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

try:
    df = pd.read_csv(r'C:\Users\pavan\OneDrive\Desktop\Training\decision_tree_regressor_dataset_v2.csv')
except FileNotFoundError:
    print("File not found. Please check the path.")
    exit()

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

df['Likelihood_Class'] = pd.qcut(df['Purchase_Likelihood'], q=3, labels=[0, 1, 2])

X = df.drop(['Purchase_Likelihood', 'Likelihood_Class'], axis=1)
y = df['Likelihood_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 15))

plt.subplot(2, 2, 1)
plot_tree(clf, feature_names=X.columns, class_names=['Low', 'Med', 'High'], filled=True, rounded=True)
plt.title("Decision Tree Structure")

plt.subplot(2, 2, 2)
feat_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values()
feat_importances.plot(kind='barh', color='skyblue')
plt.title("Feature Importance Bar Graph")


plt.subplot(2, 2, 3)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix")


plt.subplot(2, 2, 4)
top_2 = feat_importances.nlargest(2).index
X_db = X[top_2]
clf_db = DecisionTreeClassifier(max_depth=3).fit(X_db, y)

x_min, x_max = X_db.iloc[:, 0].min() - 1, X_db.iloc[:, 0].max() + 1
y_min, y_max = X_db.iloc[:, 1].min() - 1, X_db.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100),
                     np.arange(y_min, y_max, (y_max-y_min)/100))
Z = clf_db.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_db.iloc[:, 0], X_db.iloc[:, 1], c=y, edgecolor='k', cmap='RdYlBu')
plt.xlabel(top_2[0])
plt.ylabel(top_2[1])
plt.title(f"Decision Boundary ({top_2[0]} vs {top_2[1]})")

plt.tight_layout()
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))