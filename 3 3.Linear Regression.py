import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
Y = np.array([20, 30, 35, 45, 50, 55, 65, 70])

model = LinearRegression()
model.fit(X, Y)

new_hours = [[5]]
prediction = model.predict(new_hours)

print("Predicted Marks:", prediction[0])
