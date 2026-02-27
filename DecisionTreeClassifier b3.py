import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Create and encode data
df = pd.DataFrame({
    'Age': ['Youth','Youth','Middle','Senior','Senior','Senior','Middle','Youth','Youth','Senior'],
    'Income': ['High','High','High','Medium','Low','Low','Low','Medium','Low','Medium'],
    'Weekend': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','No'],
    'Buy': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes']
})

# Encode
for col in df:
    df[col] = LabelEncoder().fit_transform(df[col])

# Train
X, y = df[['Age','Income','Weekend']], df['Buy']
model = DecisionTreeClassifier().fit(X, y)

# Draw tree diagram
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=['Age','Income','Weekend'], 
          class_names=['Not Buy','Buy'], filled=True, rounded=True)
plt.show()

# Fix: Use DataFrame for prediction to avoid warning
new_data = pd.DataFrame([[1,2,1]], columns=['Age','Income','Weekend'])
print('Buy' if model.predict(new_data)[0] == 1 else 'Not Buy')
