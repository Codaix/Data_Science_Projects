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

from sklearn.metrics import r2_score #for getting the r2 score


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

#train test model:
x = df[['mass']]
y = df['orbital_period']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

#linear regression on the model:
model = LinearRegression()
model.fit(x_train,y_train)

#predicting the model:='
y_pred = model.predict(x_test)

#Evaluate:
print('R2 score: ', r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))

#visualisation fro the test set:
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, model.predict(x_test), color = 'red', linewidth = 3)
plt.xlabel("Mass")
plt.ylabel("Orbital_Period")
plt.title("Predicted Model")
plt.show()


##----- Completed -----##





