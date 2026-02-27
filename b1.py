import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Training data: [study hours, attendance %]
X = [
    [1,50],[2,55],[2.5,60],[3,58],[3.5,62],
    [4,65],[4.5,68],[5,70],[5.5,72],[6,75],
    [2,80],[3,82],[4,85],[5,88],[6,90],
    [6.5,92],[7,95],[7.5,96],[8,97],[9,98]
]

# Labels
y = ["FAIL","FAIL","FAIL","FAIL","FAIL",
     "FAIL","FAIL","FAIL","FAIL","FAIL",
     "PASS","PASS","PASS","PASS","PASS",
     "PASS","PASS","PASS","PASS","PASS"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

pred = model.predict([[5,80]])
print("Result:", pred[0])

# Scatter plot
for i in range(len(X)):
    if y[i] == "PASS":
        plt.scatter(X[i][0], X[i][1], marker='o')
    else:
        plt.scatter(X[i][0], X[i][1], marker='x')

plt.scatter(5,80, marker='*')  # new data point
plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("KNN Student Result")
plt.show()
