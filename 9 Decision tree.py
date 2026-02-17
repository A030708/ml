import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "AgeGroup":["Young","Young","Young","Middle","Middle","Old","Old","Old"],
    "IncomeLevel":["High","Low","Low","High","Low","High","Low","Low"],
    "Weekend":["Yes","Yes","No","Yes","No","Yes","No","Yes"],
    "Buy":["Buy","Buy","Not Buy","Buy","Not Buy","Buy","Not Buy","Not Buy"]
})

enc = {c: LabelEncoder().fit(df[c]) for c in df.columns}
df = df.apply(lambda x: enc[x.name].transform(x))

X, y = df.drop("Buy", axis=1), df["Buy"]
model = DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(X, y)

plt.figure(figsize=(8,5))
plot_tree(model, feature_names=X.columns, class_names=enc["Buy"].classes_, impurity=False)
plt.show()

new = pd.DataFrame({"AgeGroup":["Middle"],"IncomeLevel":["Low"],"Weekend":["Yes"]})
new = new.apply(lambda x: enc[x.name].transform(x))

print("Prediction:", enc["Buy"].inverse_transform(model.predict(new))[0])
