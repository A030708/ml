import pandas as pd
import numpy as np

data = {
    "ID":[1,2,3,4,5,6,7,8,9,10,11],
    "Name":["Alice","Bob","Charlie",None,"David","Eve","Alice","Frank","Grace","Charlie","Charlie"],
    "Age":[25,29,28,np.nan,32,35,25,100,26,28,28],
    "Salary":[50000,60000,55000,70000,None,80000,50000,100000,65000,60000,60000],
    "Department":["HR","IT","IT","HR","IT","Sales","HR","HR","IT","IT","IT"],
    "Join Date":["01-01-2020","15-05-2020","20-03-2021","25-11-2020",
                 "30-08-2021",None,"01-01-2020","18-06-2022",
                 "11-07-2021","20-03-2021","20-03-2021"],
    "Bonus":[5000,6000,5500,7000,None,8000,5000,10000,6500,5500,6000]
}

df = pd.DataFrame(data)

df["Name"] = df["Name"].fillna("Unknown")
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
df["Bonus"] = df["Bonus"].fillna(df["Bonus"].median())
df["Join Date"] = df["Join Date"].ffill()

df["Join Date"] = pd.to_datetime(df["Join Date"], format="%d-%m-%Y")

df = df.drop_duplicates()

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["Age"] >= lower) & (df["Age"] <= upper)]

print(df)
