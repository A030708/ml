import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X = np.array([[150,22],[155,25],[158,28],[160,30],[162,35],[148,20],[152,27],[157,32],[159,26],[161,29],
              [168,22],[170,25],[172,28],[175,30],[178,35],[180,40],[182,33],[176,27],[174,29],[169,24]])
y = np.array([0]*10 + [1]*10)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = SVC(kernel='linear').fit(X, y)
new = scaler.transform([[165, 26]])

print("Prediction:", "Man" if model.predict(new)[0] else "Woman")