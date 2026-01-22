import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df_raw = pd.read_csv('decision_tree_regressor_dataset_v2.csv')
df = df_raw.copy()
df.columns = ['Age', 'Gender', 'Monthly Income', 'Brand Awareness', 'Store Experience', 
              'Quality Rating', 'Price Sensitivity', 'Purchase']
df = df.apply(pd.to_numeric, errors='coerce')
df['Gender'] = df['Gender'].fillna(0).map({'Male': 0, 'Female': 1}).fillna(0).astype(int)
df['Purchase'] = df['Purchase'].fillna(0).astype(int)


X = df.drop('Purchase', axis=1)
y = df['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
