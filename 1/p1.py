import pandas as pd 
import numpy as np 
df = pd.DataFrame({ 
'EmployeeID':[101,102,103,104,105,106,107,108,101,109], 
'Name':['John','Alice','Bob','Eva','Mark','Sam','Lily','Tom','John','Riya'], 
'Age':[25,30,22,np.nan,29,35,28,40,25,np.nan], 
'Salary':[50000,60000,400000,52000,-10000,58000,62000,70000,50000,45000], 
'Department':['HR','IT','IT','Sales','IT','HR','IT','Sales','HR','IT'] 
}) 
df = df.drop_duplicates() 
df['Age'].fillna(df['Age'].mean(), inplace=True) 
df = df[df['Salary'] > 0] 
Q1,Q3 = df['Salary'].quantile([0.25,0.75]) 
IQR = Q3 - Q1 
df = df[(df['Salary'] >= Q1-1.5*IQR) & (df['Salary'] <= Q3+1.5*IQR)] 
 
print(df)
