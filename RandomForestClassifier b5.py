from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Training data: [attendance %, study hours, sleep hours]
X = [
    [60,2,5],[65,3,5],[70,3,6],[75,4,6],[80,4,6],
    [85,5,7],[90,6,7],[95,7,7],[50,2,4],[55,2,5],
    [78,5,6],[82,6,7],[88,6,7],[92,7,8],[96,8,8]
]

y = ["FAIL","FAIL","FAIL","FAIL","PASS",
     "PASS","PASS","PASS","FAIL","FAIL",
     "PASS","PASS","PASS","PASS","PASS"]

# Train Random Forest
model = RandomForestClassifier(n_estimators=5, random_state=0)
model.fit(X, y)

# Prediction
pred = model.predict([[85,5,7]])
print("Result:", pred[0])

# Plot one tree from the forest
plt.figure(figsize=(8,6))
plot_tree(model.estimators_[0],
          feature_names=['attendance','study_hour','sleep'],
          class_names=['FAIL','PASS'],
          filled=True)
plt.show()
