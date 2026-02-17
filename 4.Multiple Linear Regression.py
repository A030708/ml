import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1, 50], [2, 60], [3, 65], [4, 70],
    [5, 75], [6, 80], [7, 90]
])

Y = np.array([2, 3, 4, 5, 6, 8, 10])

model = LinearRegression()
model.fit(X, Y)

new_employee = [[4, 72]]
prediction = model.predict(new_employee)

print("Predicted Salary (lakhs):", prediction[0])
