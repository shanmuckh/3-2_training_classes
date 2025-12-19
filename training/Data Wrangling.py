import pandas as pd
import numpy as np


df_raw = pd.read_csv('decision_tree_regressor_dataset_v2.csv')
print("Raw data shape:", df_raw.shape)
print("Raw columns:", df_raw.columns.tolist())
print("First row preview:", str(df_raw.iloc[0]).split('\n')[0][:100] + "...")


df = df_raw.copy()
df.columns = ['Age', 'Gender', 'Monthly Income', 'Brand Awareness', 'Store Experience', 
              'Quality Rating', 'Price Sensitivity', 'Purchase']


df = df.apply(pd.to_numeric, errors='coerce')


print("\n1. First 5 rows:")
print(df.head())


df['Gender'] = df['Gender'].fillna(0).map({'Male': 0, 'Female': 1}).fillna(0).astype(int)


df['Purchase'] = df['Purchase'].fillna(0).astype(int)


print("\n4. Missing values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())


print("Dataset shape:", df.shape)
