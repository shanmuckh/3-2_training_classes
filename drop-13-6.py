import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 45],
        'Grade': ['A', 'B', 'C', 'D', 'E']}

df = pd.DataFrame(data)

df_dropped = df.drop(df.columns[-1], axis=1)

result = df_dropped.head(5)  

print(result)