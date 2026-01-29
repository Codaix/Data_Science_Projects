# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 16:26:40 2025

@author: PRANAV SHARMA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:/Downloads/ecommerce_customers_unit1 (1).csv")
df.head()
print("Shape :", df.shape)
print("\nData types: ")
print(df.dtypes)

print("nBasic Stats (numeric):")
print(df.describe())

missing_counts = df.isna().sum().sort_values(ascending = False)
print(missing_counts)

plt.figure(figsize=(8,4))
missing_counts.plot(kind = "bar")
plt.title("Rising Values per Column")
plt.xlabel("Columns")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

before = df.shape[0]
df= df.drop_duplicates()
after = df.shape[0]
print(f"Removed {before - after} duplicate rows. New shape: {df.shape}")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

for c in num_cols:
    if df[c].isna().any():
        med = df[c].median()
        df[c].fillna(med, inplace=True)
    
for c in cat_cols:
    if df[c].isna().any():
        mode_val = df[c].mode(dropna = True)
        if len(mode_val) > 0:
            df[c].fillna(mode_val[0], inplace = True)
        else:
            df[c].fillna("Unknown", inplace = True)

df.isna().sum().sort_values(ascending = False)

q1 = df["total_spent"].quantile(0.25)
q3 = df["total_spent"].quantile(0.75)
iqr = q3-q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print("IQR bounds: ", lower, upper)

plt.figure(figsize=(6,4))
plt.boxplot(df["total_spent"].dropna(), vert = True)
plt.title("Boxplot: total_spent")
plt.ylabel("total_spent")
plt.tight_layout()
plt.show()