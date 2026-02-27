import pandas as pd
import numpy as np

df = pd.DataFrame({
    'ID':[1,2,3,4,5,6,7,8,9,10,11],
    'Name':['Alice','Bob','Charlie','David','Eve','Alice','Frank','Grace','Charlie','Charlie','Charlie'],
    'Age':[25,29,28,np.nan,32,35,25,100,26,28,28],
    'Salary':[50000,60000,55000,70000,65000,80000,50000,100000,65000,60000,60000],
    'Department':['HR','IT','IT','HR','IT','Sales','HR','HR','IT','IT','IT'],
    'Bonus':[5000,6000,5500,7000,np.nan,8000,5000,10000,6500,5500,6000]
})

df = df.drop_duplicates()

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Bonus'] = df['Bonus'].fillna(df['Bonus'].mean())

Q1, Q3 = df['Age'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[df['Age'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)]

Q1, Q3 = df['Salary'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[df['Salary'].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)]

df.reset_index(drop=True, inplace=True)

print(df)
