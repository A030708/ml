import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = [[1,2],[2,3],[3,3],[8,7],[8,8],[7,7],
     [20,20],[21,22],[22,21],[1,3],[2,2],
     [7,8],[8,6],[21,21],[22,22]]

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=[labels[i]])

plt.scatter(centers[:,0], centers[:,1], marker='*', s=200)
plt.title("K-Means Clustering")
plt.show()
