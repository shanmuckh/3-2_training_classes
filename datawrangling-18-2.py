import pandas as pd
import numpy as np

data = {
    'Name': ['Amit', 'Neha', 'Rahul', 'Pooja'],
    'Age': [25, np.nan, 28, np.nan],
    'Salary': [6000, 5000, np.nan, 55000]
}

df = pd.DataFrame(data)

print("Missing values count:")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

print("\nUpdated DataFrame:")
print(df)