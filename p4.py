import numpy as np
from sklearn.linear_model import LinearRegression

hours = np.array([1,2,3,4,5,6,7,8,2.5,3.5,4.5,5.5,6.5,7.5,8.5]).reshape(-1,1)
marks = np.array([35,40,45,50,55,60,65,70,42,48,52,58,63,68,72])

lr = LinearRegression()
lr.fit(hours, marks)
print("Marks for 6 hours:", round(lr.predict([[6]])[0],2))


experience = [1,2,3,4,5,6,7,8,2,3,4,5,6,7,8]
education = [12,12,12,15,15,15,16,16,12,12,15,15,16,16,18]
salary = [20000,25000,30000,35000,40000,45000,50000,55000,
          26000,31000,36000,41000,46000,51000,60000]

X = np.array(list(zip(experience, education)))

mlr = LinearRegression()
mlr.fit(X, salary)
print("Salary:", round(mlr.predict([[5,16]])[0],2))
