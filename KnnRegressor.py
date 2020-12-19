#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:45:17 2020

@author: Waquar Shamsi
"""

# KNN Regressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

stock = 'WIPRO'
file_stock = 'Dataset/' + stock + '.csv'
stock_df = pd.read_csv(file_stock)
stoc_df = stock_df.drop(['Deliverable Volume','%Deliverble','Turnover','Trades','Volume','VWAP','Symbol','Series'],axis=1)
X = stoc_df[['Prev Close','Open','High','Low','Last']]
y = stoc_df['Close']

#X_Train has last 6 months data - 7month=210 days - 30 days = 180 days
X_train = X.tail(210).head(180)
y_train = y.tail(210).head(180)
X_test = X.tail(30)
y_test = y.tail(30)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,y_train)
pred_knn = model.predict(X_test)

plt.figure(figsize=(15,10))
plt.plot(list(range(1,181)), y_train, color='blue')
plt.plot(list(range(181,211)), y_test, color='green',label='actual')
plt.plot(list(range(181,211)), pred_knn, color='red',label='predicted')
dates = stoc_df['Date'].tail(210).to_numpy()[1::10]
plt.locator_params(nbins=6,axis='x')
plt.xlabel = 'Days'
plt.ylabel = 'Stock Price'
plt.legend()
plt.show()

print('mse',mean_squared_error(y_test, pred_knn))
print('score',r2_score(y_test, pred_knn))
