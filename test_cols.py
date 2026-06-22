import pandas as pd
df = pd.read_csv("data/cleaned_dataset.csv", nrows=100)
categorical_cols = df.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
print(categorical_cols)
for c in categorical_cols:
    print(c, df[c].nunique())
