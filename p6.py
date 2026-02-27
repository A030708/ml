from sklearn.tree import DecisionTreeClassifier

# Training data: [Income, Credit Score, Loan Amount]
X = [
    [25000,650,200000],
    [30000,700,150000],
    [40000,750,200000],
    [50000,800,250000],
    [35000,720,180000],
    [28000,680,220000],
    [45000,770,240000],
    [32000,690,200000],
    [52000,820,260000],
    [29000,640,210000],
    [41000,760,230000],
    [36000,710,190000],
    [48000,790,250000],
    [33000,705,200000],
    [55000,830,270000]
]

# 1 = Approve, 0 = Reject
y = [0,1,1,1,1,0,1,0,1,0,1,1,1,1,1]

model = DecisionTreeClassifier()
model.fit(X, y)

prediction = model.predict([[37000,720,200000]])

print("Loan Status:", "APPROVED" if prediction[0] == 1 else "REJECTED")
