# Project idea by codeAcademy
# Project done by Artur Wiktorowicz
# 20/10/2021

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("data/honeyproduction.csv")

# Storing mean production value by year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Storing years in a vector form
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# Storing production data in a vector form
y = prod_per_year['totalprod']

# Creating Linear Regression Model 
regr = linear_model.LinearRegression()
regr.fit(X, y)
y_predict = regr.predict(X)

# Predicting honey production for future years
X_future = np.array(range(2013,2050))
X_future = X_future.reshape(-1,1)
future_predict = regr.predict(X_future)

# Visuilation
plt.scatter(X, y)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.show()