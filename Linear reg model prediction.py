# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:20:01 2025

@author: PRANAV SHARMA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



x = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,5,4,5])
#splitting data in 20% test and 80% training
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
#radnom state used for code reproducability



model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")



plt.scatter(x,y,color = 'blue', label='Actual Data')
plt.plot(x, model.predict(x), color = 'red',linewidth = 2, label='Regression Line')
plt.xlabel("House Size (sq ft")
plt.ylabel("price ($1000s")
plt.title("Linear regression: House Size vs. Price")
plt.legend()
plt.grid(True)
plt.show()



a = np.array([[3500]])
predicted_price = model.predict(a)
print(predicted_price)
