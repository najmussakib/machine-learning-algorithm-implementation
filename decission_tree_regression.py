# Decision Tree Regression  is a non linear and non continuous
"""
Created on Mon Aug  5 23:01:47 2019

@author: Najmus Saqib
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Creating the matrix of independent and dependent features
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
'''
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# Predicting the new results
y_pred = regressor.predict(x)

'''# Visualising the Decission Tree Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (Decission Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''

# Visualising the Decission Tree Regression results(for higher resolution and smoother curve) because DT is a non continuous curve
x_grid = np.arange(min(x) , max(x) ,0.01)
x_grid = x_grid.reshape((len(x_grid) , 1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid , regressor.predict(x_grid) , color = 'blue')
plt.title('Truth or Bluff (Decission Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#the plot show the result of a single tree
