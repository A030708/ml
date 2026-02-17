import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[2,55],[3,60],[3,65],[4,70],[4,68],[5,72],[5,75],[6,78],[6,82],[7,85],[7,88],[8,90]])
y = np.array([0,0,0,0,0,1,1,1,1,1,1,1])

knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
new = np.array([[5,76]])

print("Prediction:", "Pass" if knn.predict(new)[0] else "Fail")

plt.scatter(X[:,0], X[:,1], c=y)
plt.scatter(new[:,0], new[:,1], marker='X', s=120)
plt.show()
