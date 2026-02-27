import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

hours = np.array([1,2,3,4,5,6,7,8,2,3,4,5,6,7,8]).reshape(-1,1)
result = np.array([0,0,0,0,1,1,1,1,0,0,1,1,1,1,1])

nb = GaussianNB().fit(hours, result)
lr = LogisticRegression().fit(hours, result)

print("Naive Bayes:", "Pass" if nb.predict([[5]])[0] else "Fail")
print("Logistic Regression:", "Pass" if lr.predict([[5]])[0] else "Fail")
