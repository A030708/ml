import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.array([[1,30],[2,35],[2,40],[3,42],[5,60],[6,62],[6,65],[7,68],[9,85],[10,88],[11,90],[12,95]])
k = KMeans(n_clusters=3, random_state=42).fit(data)

new = [[6,64]]
print("Cluster:", k.predict(new)[0])

plt.scatter(data[:,0], data[:,1], c=k.labels_)
plt.scatter(*k.cluster_centers_.T, marker='X', s=150)
plt.scatter(*np.array(new).T, marker='*', s=200)
plt.show()
