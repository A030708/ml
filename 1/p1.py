import pandas as pd
import numpy as np

data = {
    "Name": ["Alice","Bob","Charlie",np.nan,"David","Eve","Alice","Frank","Grace","Charlie","Charlie"],
    "Age": [25,29,28,np.nan,32,35,25,100,26,28,28],
    "Salary": [50000,60000,55000,70000,np.nan,80000,50000,100000,65000,60000,60000],
    "Department": ["HR","IT","IT","HR","IT","Sales","HR","HR","IT","IT","IT"],
    "Bonus": [5000,6000,5500,7000,np.nan,8000,5000,10000,6500,5500,6000]
}

df = pd.DataFrame(data)

# Fill missing values
df["Name"] = df["Name"].fillna("Unknown")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Bonus"] = df["Bonus"].fillna(0)

# Remove duplicates
df = df.drop_duplicates()

# Remove outliers
df = df[df["Age"] <= 60]

print(df)