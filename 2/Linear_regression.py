import numpy as np
from sklearn.linear_model import LinearRegression

# 15 years data
year = np.array([2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]).reshape(-1,1)
rainfall = np.array([820,830,780,790,800,860,870,890,900,910,920,940,950,970,980])

model = LinearRegression()
model.fit(year, rainfall)

# Predict rainfall for 2020
pred = model.predict([[2020]])

print("Predicted Rainfall in 2020:", pred[0])