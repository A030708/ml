import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([
    [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],
    [6, 8], [7, 9], [8, 9], [9, 10]
])

Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

model = GaussianNB()
model.fit(X, Y)

new_student = [[1, 2]]
prediction = model.predict(new_student)

print("Prediction:", "PASS" if prediction[0] == 1 else "FAIL")
