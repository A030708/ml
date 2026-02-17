import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([
    [2, 300], [3, 320], [4, 330],
    [5, 400], [6, 420], [7, 450]
])

Y = np.array([0, 0, 0, 1, 1, 1])

model = DecisionTreeClassifier()
model.fit(X, Y)

new_applicant = [[5, 390]]
prediction = model.predict(new_applicant)

print("Loan Status:", "APPROVED" if prediction[0] == 1 else "REJECTED")
