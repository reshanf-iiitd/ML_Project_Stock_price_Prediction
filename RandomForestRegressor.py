#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:43:55 2020

@author: Waquar Shamsi
"""

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

stock = 'WIPRO'
file_stock = 'Dataset/' + stock + '.csv'
stock_df = pd.read_csv(file_stock)
stoc_df = stock_df.drop(['Deliverable Volume','%Deliverble','Turnover','Trades','Volume','VWAP','Date','Symbol','Series','Last'],axis=1)

X = stoc_df[['Open','High','Low']]
y = stoc_df['Close']

X_train = X.tail(210).head(180)
y_train = y.tail(210).head(180)
X_test = X.tail(30)
y_test = y.tail(30)

regr = RandomForestRegressor(max_depth=10)
regr.fit(X_train,y_train)
pred_dt = regr.predict(X_test)

plt.figure(figsize=(15,10))
plt.plot(list(range(1,181)), y_train, color='blue')
plt.plot(list(range(181,211)), y_test, color='green',label='actual')
plt.plot(list(range(181,211)), pred_dt, color='red',label='predicted')
plt.xlabel = 'Days'
plt.ylabel = 'Stock Price'
plt.legend()
plt.show()
print('mse',mean_squared_error(y_test, pred_dt))
print('score',r2_score(y_test, pred_dt))
