import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    "Attendance":["High","High","High","Medium","Medium","Medium","Low","Low","Low","High","Low","Low"],
    "Study":["High","Medium","Low","High","Medium","Low","High","Medium","Low","Medium","High","High"],
    "Sleep":["Good","Good","Poor","Good","Poor","Poor","Good","Poor","Poor","Poor","Poor","Poor"],
    "Result":["Pass","Pass","Fail","Pass","Fail","Fail","Pass","Fail","Fail","Pass","Pass","Fail"]
})

le = {c: LabelEncoder().fit(df[c]) for c in df}
df = df.apply(lambda x: le[x.name].transform(x))

X, y = df.drop("Result", axis=1), df["Result"]
rf = RandomForestClassifier(n_estimators=5, max_depth=3).fit(X, y)

plot_tree(rf.estimators_[0], feature_names=X.columns,
          class_names=le["Result"].classes_, filled=True)
plt.show()
