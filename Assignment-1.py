import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

data = {
    'Age': [58, 48, 34, 27, 40, 58],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Male', 'Female'],
    'Monthly_Income': [81476, 64811, 56208, 40150, 91180, 63286],
    'Brand_Awareness': [2, 2, 2, 3, 3, 2],
    'Store_Experience': [1, 4, 3, 4, 4, 4],
    'Quality_Rating': [3, 4, 4, 5, 2, 3],
    'Price_Sensitivity': [3, 5, 1, 5, 2, 2],
    'Purchase_Likelihood': [1.9325, 1.9359, 2.3758, 2.0955, 1.9647, 1.5230]
}

df = pd.DataFrame(data)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Purchase'] = (df['Purchase_Likelihood'] > df['Purchase_Likelihood'].median()).astype(int)

X = df.drop(['Purchase_Likelihood', 'Purchase'], axis=1)
y = df['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr = LogisticRegression().fit(X_train, y_train)
dt = DecisionTreeClassifier(criterion='gini', random_state=42).fit(X_train, y_train)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.histplot(df['Monthly_Income'], color='green', kde=True)
plt.title('Monthly Income Distribution')

plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, dt.predict(X_test))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')

plt.subplot(2, 1, 2)
plot_tree(dt, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree Structure')

plt.tight_layout()
plt.show()
