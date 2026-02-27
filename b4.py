from sklearn.neural_network import MLPClassifier

# XOR inputs and outputs
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]

model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=5000)
model.fit(X, y)

print("0 XOR 0 =", model.predict([[0,0]])[0])
print("0 XOR 1 =", model.predict([[0,1]])[0])
print("1 XOR 0 =", model.predict([[1,0]])[0])
print("1 XOR 1 =", model.predict([[1,1]])[0])
