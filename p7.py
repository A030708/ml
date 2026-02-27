from sklearn import svm

# Training data: [height(cm), age]
X = [
    [150,20],[152,22],[155,24],[158,26],[160,28],
    [162,30],[165,32],[168,34],[170,36],[172,38],
    [155,21],[158,23],[160,25],[165,27],[168,29],
    [170,31],[172,33],[175,35],[178,37],[180,39]
]

# Labels
y = ["Woman","Woman","Woman","Woman","Woman",
     "Woman","Woman","Woman","Man","Man",
     "Woman","Woman","Woman","Man","Man",
     "Man","Man","Man","Man","Man"]

# Train model
model = svm.SVC()
model.fit(X, y)

# Predict
pred = model.predict([[190,28]])

print("Prediction:", pred[0])

