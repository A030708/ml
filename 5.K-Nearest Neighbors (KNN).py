import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [2, 20], [3, 25], [4, 30],
    [5, 40], [6, 42], [7, 45],
    [8, 50], [9, 55], [10, 60]
])

Y = np.array(["C", "C", "C", "B", "B", "B", "A", "A", "A"])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, Y)

new_student = [[6, 50]]
prediction = model.predict(new_student)

print("Predicted Grade:", prediction[0])