#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 00:01:47 2020

@author: Waquar Shamsi
"""

# RandomForest Regressor with Sentiment Analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import r2_score
import pickle
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
test_dates = stock_df[['Date']]
test_dates = test_dates.tail(30).values.tolist()

regr = RandomForestRegressor(max_depth=10)
regr.fit(X_train,y_train)
pred_dt = regr.predict(X_test)



test_news_file = open('test_news_data', 'rb') 
test_news = pickle.load(test_news_file)

i=0
for stock_date in test_dates:
  dated_df = test_news.loc[test_news['Date'] == stock_date[0]]
  polarity = dated_df['Polarity'].sum(axis = 0) 
  print(polarity)
  if polarity > 0:
    #positive sentiment
    pred_dt[i]=pred_dt[i]*(((polarity/100)*0.5)+1) # increase by 10%
  elif polarity < 0:
    #negative sentiment
    pred_dt[i]=pred_dt[i]*0.9 # decrease by 10%
  #for neutral sentiment - no change
  i+=1



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

