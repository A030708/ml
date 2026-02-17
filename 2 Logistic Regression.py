import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [2, 40], [3, 45], [4, 50], [5, 55], [6, 60],
    [7, 70], [8, 75], [9, 80], [10, 85]
])

Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, Y)

new_student = [[6, 65]]
prediction = model.predict(new_student)

print("Prediction:", "PASS" if prediction[0] == 1 else "FAIL")
