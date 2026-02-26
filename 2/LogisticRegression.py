import numpy as np
from sklearn.linear_model import LogisticRegression

# 15 student data
hours = np.array([1,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,9]).reshape(-1,1)
result = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])

model = LogisticRegression()
model.fit(hours, result)

# Predict result if student studies 4 hours
pred = model.predict([[4]])

print("Pass(1) or Fail(0):", pred[0])