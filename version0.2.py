# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:05:58 2020

@author: sunny jaiswal
"""

"""
Life expectancy using linear regression
version 0.2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""creating a data frame"""
df = pd.read_csv('Life Expectancy Data.csv')

"""EDA"""
"""Dimensions of dataframe """
print(df.shape)

"""To print list of all attributes/coulmns"""
print(df.columns)

"""To present the info about dataframe"""
print(df.info)

"""counting the no. of null values"""
print(df.isnull().sum())

"""Repalcing null values with median"""
df.fillna(df.median(),inplace = True)
print(df.isnull().sum())

#pre_drop = df.shape[0]
#df = df.dropna()
#post_drop = df.shape[0]
#
#print("no. of rows before dropping nullvalues",pre_drop)
#print("no. of rows after dropping nullvalues",post_drop)

print(df.describe().transpose())

"""Plotting correlations among variables of dataset"""
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

"""Life expectancy just on GDP"""
y = df['Life expectancy '].values
X = df.loc[:,['Status','GDP']]
y=y.reshape(-1,1)

"""Replacing categorical varibales with integer values"""
X.Status = pd.Categorical(X.Status,['Developing','Developed'],ordered=True)
X.Status = X.Status.cat.codes
print(X)

"""scatter plot of y vs X_gdp"""
plt.figure(figsize=(10,8))
plt.scatter(X['GDP'], y)
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.show()

"""scatter plot of y vs Status"""
plt.figure(figsize=(10,8))
plt.scatter(X['Status'], y)
plt.xlabel('Status')
plt.ylabel('Life Expectancy')
plt.show()

"""Spitting dataset into training set and test set"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 1/10,random_state = 0)

"""fitting Simple linear regression model on training set"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) 

"""predicting the test set results"""
y_pred = regressor.predict(x_test) #y_pred is predicted vector set of x_test matrix

"""K-cross validation"""
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor,X=x_train,y=y_train,cv=20)
"""Visualising the Training set results
plt.figure(figsize=(10,8))
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Life expectancy vs GDP(Training set)')
plt.xlabel('GDP')
plt.ylabel('Life expectancy')
plt.show()"""

"""Visualising the Test set results
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_test,regressor.predict(x_test),color = 'blue')
plt.title('Life expectancy vs GDP(Test set)')
plt.xlabel('GDP')
plt.ylabel('Life expectancy')
plt.show()"""

"""Calculating R-square value"""
def r_square(y_pred,y_test,y):
	y_mean = np.mean(y)
	print('Life expectancy mean value:',y_mean)
	t1=t2=0
	for i in y_pred:
		t1 += ((i-y_mean)**2)
	print(t1)
	for j in y_test:
		t2 += ((j-y_mean)**2)
	print(t2)
	return t1/t2

"""Test the model"""
gdp = float(input("Enter a GDP Value for any country:"))
status = float(input("Enter Status value of country:"))
a=np.array([status,gdp])
a.resize((1,2))
print(a)

life_exp = regressor.predict(a)
print("Predicted life expectancy : ",life_exp)

"""Model Evaluation"""
print('R-squared value ',r_square(y_pred,y_test,y))
from sklearn.metrics import mean_absolute_error
print('MAE : ',mean_absolute_error(y_test,y_pred))
print('RMSE : ',np.sqrt(np.mean((y_pred-y_test)**2)))



  