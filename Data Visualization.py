import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_raw = pd.read_csv('decision_tree_regressor_dataset_v2.csv')
df = df_raw.copy()
df.columns = ['Age', 'Gender', 'Monthly Income', 'Brand Awareness', 'Store Experience', 
              'Quality Rating', 'Price Sensitivity', 'Purchase']
df = df.apply(pd.to_numeric, errors='coerce')
df['Gender'] = df['Gender'].fillna(0).map({'Male': 0, 'Female': 1}).fillna(0).astype(int)
df['Purchase'] = df['Purchase'].fillna(0).astype(int)

print("Dataset loaded for visualization!")
print("Shape:", df.shape)


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.hist(df['Monthly Income'], bins=20, edgecolor='black', alpha=0.7)
plt.title('1. Monthly Income Histogram')
plt.xlabel('Monthly Income')


plt.subplot(1, 3, 2)
df['Purchase'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('2. Purchase Distribution')
plt.xlabel('Purchase (0=NO, 1=YES)')
plt.xticks(rotation=0)


plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Brand Awareness', y='Quality Rating', hue='Purchase', alpha=0.6)
plt.title('3. Brand Awareness vs Quality Rating')

plt.tight_layout()
plt.show()


