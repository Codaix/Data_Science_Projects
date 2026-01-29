# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 09:13:55 2025

@author: PRANAV SHARMA
"""

# Polynomial Regression

# Extension of Linear Regression used to find relationship between dependent and independent variables
# Equation: Y = b0 + b1X + b2x^2 +..... + bnX^n
# Linear in Coefficieants/slope/parameters but not in x

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.array([800, 1000, 1200, 1500, 1800]).reshape(-1, 1)
y = np.array([45,50,55,70,85])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(X_poly, y)

print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2):", model.coef_)

#Predict
x_new = np.linspace(800, 1800, 100).reshape(-1,1)
y_pred = model.predict(poly.transform(x_new))

#Plot
plt.scatter(x,y,color='blue')
plt.plot(x_new, y_pred, color = 'green')
plt.xlabel("Size (sqft)")
plt.ylabel("Price (lakhs)")
plt.title("Polynomial Regression (Degree 2)")
plt.show()