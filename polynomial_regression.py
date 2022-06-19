#Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Creating matrix of features (independent variable )
x = dataset.iloc[:, 1:2].values #making this a matrix instead of a vector
 
# Creating matrix of features (dependent variable (profit))
y = dataset.iloc[:, 2].values
 

# Splitting The data set into the training and test set
#dont need to split as dataset is very small
'''from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2 ,random_state=0) #20% is test data
'''

#Fitting Linear regression to  the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Fitting Polynomial regression to  the dataset
#Polynomiafeatures class transfornms a matix of features into the matrix of there squares
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)   #degree arg implies how many columns of squares we want the matrix transform
x_poly = poly_reg.fit_transform(x)    #fit_trans() transforms x into polynomial feature matrix and saves it in X_poly
lin_reg2 = LinearRegression()   #lin_reg2 will fit x_poly into linear Regression
lin_reg2.fit(x_poly,y)

#visualising the Linear Regression results
plt.scatter(x,y ,color = 'red')    #plotting real observation points
plt.plot(x,lin_reg.predict(x),color ='blue')  #plotting predictions which were predicted by the lin_reg regressor
plt.title('Truth Or Bluff(Linear regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualising the Polynomial Regression results
x_grid = np.arange(min(x) ,max(x) ,0.1)  #making the curve smooth taking small steps(.1)
x_grid = x_grid.reshape((len(x_grid)),1)  #1 is the number of columns
plt.scatter(x,y ,color = 'red')    #plotting real observation points
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color ='blue')  #plotting predictions which were predicted by the lin_reg regressor
plt.title('Truth Or Bluff(Polynomial regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

'''#Predicting a new result with Linear Regression 
lin_reg2.predict(6.5)

#Predicting a new result with Polynomial Regression 
lin_reg2.predict(poly_reg.fit_transform(6.5))