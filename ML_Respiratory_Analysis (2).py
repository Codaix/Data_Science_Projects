import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv("/mnt/data/Respiratory_Virus_Hospital_Admissions_Over_Time.csv")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
# Time-based features
df['week'] = df['Date'].dt.isocalendar().week.astype(int)
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# Lag features (previous weeks)
df['lag_1'] = df['Admissions'].shift(1)
df['lag_2'] = df['Admissions'].shift(2)
df['lag_4'] = df['Admissions'].shift(4)

# Rolling statistics
df['rolling_mean_3'] = df['Admissions'].rolling(3).mean()
df['rolling_std_3'] = df['Admissions'].rolling(3).std()

df.dropna(inplace=True)
X = df.drop(['Date', 'Admissions'], axis=1)
y = df['Admissions']

# Keep temporal order
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)
# Surge threshold (top 25%)
threshold = df['Admissions'].quantile(0.75)

df['Surge'] = (df['Admissions'] >= threshold).astype(int)

# Predict surge from admissions forecast
df.loc[X_test.index, 'Predicted_Admissions'] = y_pred
df['Predicted_Surge'] = (df['Predicted_Admissions'] >= threshold).astype(int)

accuracy = accuracy_score(
    df.loc[X_test.index, 'Surge'],
    df.loc[X_test.index, 'Predicted_Surge']
)

print("Surge Detection Accuracy:", accuracy)
corr = df[['Admissions', 'COVID', 'RSV', 'Influenza']].corr()

print(corr['Admissions'])
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation with Hospital Admissions")
plt.show()
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Admissions'], label='Actual')
plt.plot(df.loc[X_test.index, 'Date'], y_pred, label='Predicted')
plt.axhline(threshold, color='red', linestyle='--', label='Surge Threshold')
plt.legend()
plt.title("Weekly Hospital Admission Forecast")
plt.show()
