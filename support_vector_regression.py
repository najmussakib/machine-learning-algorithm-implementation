# -*- coding: utf-8 -*-
# SVR is a non linear regression
"""
Created on Sat Aug  3 21:19:17 2019

@author: Najmus Saqib
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

#creating a matrix of features , independent and dependent var
x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]

#feature scaling have to be done because SVR doesnt apply feature scalling implicitly
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() 
sc_y = StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)
 

#fitting the svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#predicting a new result set
y_pred =sc_y.inverse_transform( regressor.predict(sc_x.transform(np.array([[x]]))))  #.transform() takes arguments as arrays  that is why 6.5 is converted to an arrar by np.array([[]]) ,single[] will make 6.5 a vector


#Visualising the SVR results
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x) ,color = 'blue')
plt.title('Truth Or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()







