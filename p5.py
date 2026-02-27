import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Training data (Height, Weight)
X = [
    [150,45],[152,48],[155,50],[157,52],[160,54],
    [162,56],[165,58],[168,60],[170,62],[172,65],
    [155,60],[158,63],[160,65],[162,67],[165,70],
    [168,72],[170,75],[172,78],[175,80],[178,85]
]

# Labels (0 = Class 0, 1 = Class 1)
y = [0,0,0,0,0,0,0,0,0,0,
     1,1,1,1,1,1,1,1,1,1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

prediction = model.predict([[165,60]])

print("Predicted Class:", prediction[0])
