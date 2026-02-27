from sklearn.tree import DecisionTreeClassifier

# Training data
X = [
    ["Young","High","Yes"],
    ["Young","Low","Yes"],
    ["Young","Low","No"],
    ["Middle","High","Yes"],
    ["Middle","Low","No"],
    ["Old","High","Yes"],
    ["Old","Low","No"],
    ["Old","Low","Yes"]
]

y = ["Buy","Buy","Not Buy","Buy","Not Buy","Buy","Not Buy","Not Buy"]

# Convert text to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X_encoded = []
for col in zip(*X):
    X_encoded.append(le.fit_transform(col))

X_encoded = list(zip(*X_encoded))

model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# New data: Middle, Low, Yes
new = [[1,0,1]]
print("Prediction:", model.predict(new)[0])
