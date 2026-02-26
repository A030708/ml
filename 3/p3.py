import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

hours = np.array([1,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,9]).reshape(-1,1)
result = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]) 

lr = LogisticRegression()
lr.fit(hours, result)
lr_pred = lr.predict([[6]])

nb = GaussianNB()
nb.fit(hours, result)
nb_pred = nb.predict([[6]])

print("Logistic Regression Prediction:", lr_pred[0])
print("Naive Bayes Prediction:", nb_pred[0])