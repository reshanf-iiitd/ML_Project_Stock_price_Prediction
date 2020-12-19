#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:47:27 2020

@author: Waquar Shamsi
"""
#LSTM
# LSTM USING LIBRARY

from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
import numpy as np

stock = 'WIPRO'
file_stock = 'Dataset/' + stock + '.csv'
stock_df = pd.read_csv(file_stock)
stoc_df = stock_df.drop(['Deliverable Volume','%Deliverble','Turnover','Trades','Volume','VWAP','Symbol','Series'],axis=1)
# X = stoc_df[['Prev Close','Open','High','Low','Last']]
# y = stoc_df['Close']

selected_df = stoc_df[['Prev Close','Open','High','Low','Last','Close']]
#X_Train has last 6 months data - 7month=210 days - 30 days = 180 days
# X_train = X.tail(210).head(180)
# y_train = y.tail(210).head(180)
# X_test = X.tail(30)
# y_test = y.tail(30)


training_set = selected_df.tail(210).head(180).to_numpy()
testing_set = selected_df.tail(30).to_numpy()
y_test = testing_set[:,5:6]
y_train = training_set[5:,5:6]
ts = training_set[:,5:6]
# print(training_set)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(ts)


X_tr = []
y_tr = []
for i in range(5, 180):
  X_tr.append(training_set_scaled[i-5:i, 0])
  y_tr.append(training_set_scaled[i, 0])
X_tr, y_tr = np.array(X_tr), np.array(y_tr)
X_tr = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], 1))


model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_tr.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_tr,y_tr,epochs=1000,batch_size=32)



real_stock_price = testing_set[:, 5:6]
dataset_train = pd.DataFrame(ts,columns=['Close'])
dataset_test = pd.DataFrame(real_stock_price,columns=['Close'])
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 5:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_te = []
for i in range(5, 35):
  X_te.append(inputs[i-5:i, 0])
X_te = np.array(X_te)
X_te = np.reshape(X_te, (X_te.shape[0], X_te.shape[1], 1))
predicted_stock_price = model.predict(X_te)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

print(y_test.shape)
print(predicted_stock_price.shape)

print('mse',mean_squared_error(y_test, predicted_stock_price))
print('score',r2_score(y_test, predicted_stock_price))



plt.figure(figsize=(15,10))
plt.plot(list(range(1,176)), y_train, color='blue')
plt.plot(list(range(181,211)), y_test, color='green',label='actual')
plt.plot(list(range(181,211)), predicted_stock_price, color='red',label='predicted')
dates = stoc_df['Date'].tail(210).to_numpy()[1::10]
plt.locator_params(nbins=6,axis='x')
plt.xlabel = 'Days'
plt.ylabel = 'Stock Price'
plt.legend()
plt.show()

