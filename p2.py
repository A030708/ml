import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

# Linear Regression
year = np.array([2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,
                 2011,2012,2013,2014,2015]).reshape(-1,1)

rain = np.array([800,820,780,790,850,870,900,920,940,960,
                 980,1000,1020,1050,1100])

lr = LinearRegression()
lr.fit(year, rain)

print("Predicted rainfall for 2016 =", lr.predict([[2016]])[0])


# Logistic Regression
hours = np.array([1,2,3,4,5,6,7,8,2,3,4,5,6,7,8]).reshape(-1,1)
result = np.array([0,0,0,0,1,1,1,1,0,0,1,1,1,1,1])

log = LogisticRegression()
log.fit(hours, result)

print("Pass/Fail for 5 study hours =", log.predict([[5]])[0])
