from sklearn.ensemble import RandomForestClassifier

# Training data: [attendance %, study hours, sleep hours]
X = [
    [60,2,5],[65,3,5],[70,3,6],[75,4,6],[80,4,6],
    [85,5,7],[90,6,7],[95,7,7],[50,2,4],[55,2,5],
    [78,5,6],[82,6,7],[88,6,7],[92,7,8],[96,8,8]
]

y = ["FAIL","FAIL","FAIL","FAIL","PASS",
     "PASS","PASS","PASS","FAIL","FAIL",
     "PASS","PASS","PASS","PASS","PASS"]

model = RandomForestClassifier()
model.fit(X, y)

pred = model.predict([[85,5,7]])

print("Result:", pred[0])
