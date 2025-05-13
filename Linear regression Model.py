import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#print(sns.get_dataset_names())

#getting the datset:
planets = sns.load_dataset("planets")

print(planets.info())
print(planets.describe())
print(planets.head())
print(planets.tail())
#making the datset
#here, the columns are taken the one having some na values soo as to revise the na handling concept :3

#df = planets[["number","orbital_period", "mass", "year"]]
df = pd.DataFrame(planets, columns=["number", "orbital_period","mass","distance", "year"])
#print(df)

#checking the outliers:
sns.boxenplot(data= df[['number', 'orbital_period', 'mass', 'distance', 'year']])
plt.show() 
#output: the orbital_period has the highest number of outliers and the others dont really have any outliers.
    
#handling not null values:
df['orbital_period']=df['orbital_period'].fillna(df['orbital_period'].median())
df['mass'] = df['mass'].fillna(df['mass'].mean())
df['distance'] = df['distance'].fillna('Unknown')
print(df.isna().sum())

#starting with Regression model
from sklearn.model_selection import train_test_split #this one's for split the data into training and testing sets

from sklearn.preprocessing import MinMaxScaler #normalises the dataset into standard normal distribution

from sklearn.linear_model import LinearRegression #actual linear regression model

from sklearn.metrics import mean_squared_error # for getting the MSE


#defining coordinates X and Y of the model where X has the independent variables and Y is the target value or
#the dependent variable for which we want to make the model.

X = df[['number', 'mass', 'year']]
Y = df[['orbital_period']]

#Normalisation
scalar = MinMaxScaler()
x_scaled = scalar.fit_transform(X) #gets the values of all column in the X field to be in between 0 to 1
 #visualise:
sns.histplot(x_scaled, bins = 100, kde = True)
plt.title("X_scaled visualised")
plt.xlabel("x_scaled")
plt.ylabel("values of the columns selected")
plt.show()

#next step is to -> train test -> train the linear regression model -> predict on test data-> Evaluate the model

##----- yet to be completed. -----##





