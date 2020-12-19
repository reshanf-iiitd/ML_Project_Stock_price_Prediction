
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.style.use('bmh')
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import max_error
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.simplefilter("ignore")


# In[2]:


# function to predict arima and also calculating accuracy
def arima_model(name):
    #reading data from csv and making date column as index
    df = pd.read_csv(name, index_col='Date',parse_dates=True)       
    #selecting previous closing value for prediction
    price = df['Prev Close']

    #plotting and saving graph of whole closing price
    name = name.split('.')[0]
    s = name + '_orignal_closing_price'
    price.plot(label=s, legend=True, figsize=(12,5))
    plt.title(s)
    s = s + '.png'
    plt.savefig(s)
    plt.show()

    #finding the best order of arima
    stepwise_fit = auto_arima(df['Prev Close'], trace=True, suppress_warnings=True)
    
    #spliting data into train and test and chossing last 120 days for prediction
    train = price.iloc[:-120]
    test = price.iloc[-120:]

    #applying arima model and printing its summary
    model=ARIMA(price,order=stepwise_fit.order)
    model=model.fit()
    print(model.summary())

    #predicting the values using arima for last 120days
    a=len(train)
    b=len(train)+len(test)-1    
    pred=model.predict(start=a,end=b,typ='levels')

    #plotting and saving graph of test vs predictions
    s = name + '_ARIMA_Predictions'
    pred.plot(label= s, legend=True)
    s = name + '_Actual_Data'
    test.plot(label=s, legend=True)
    s = name + '_armima_test_Comparision'
    plt.title(s)
    s = name + '_armima_test_Comparision.png'
    plt.savefig(s)
    plt.show()

    #plotting and saving graph of whole data and predictions
    s = name + '_Actual_Data'
    price.plot(label=s, legend=True, figsize=(12,5))
    s = name + '_ARIMA_Predictions'
    pred.plot(label= s, legend=True)
    s = name + '_armima_all'
    plt.title(s)
    s = name + '_armima_all.png'
    plt.savefig(s)
    plt.show()

    #calculating and printing errors between test and pred
    print('Mean Squared Error: ',mean_squared_error(test, pred))
    print('Mean Squared Log Error: ',mean_squared_log_error(test, pred))
    print('Mean Absolute Error: ',mean_absolute_error(test, pred))
    print('Max Error',max_error(test, pred))


# In[7]:


# function to predict sarima and also calculating accuracy
def sarima_model(name):
    #reading data from csv and making date column as index
    df = pd.read_csv(name, index_col='Date',parse_dates=True)       
    #selecting previous closing value for prediction
    price = df['Prev Close']

    #plotting and saving graph of whole closing price
    name = name.split('.')[0]
    s = name + '_orignal_closing_price'
    price.plot(label=s, legend=True, figsize=(12,5))
    plt.title(s)
    s = s + '.png'
    plt.savefig(s)
    plt.show()

    #finding the best order of arima
    stepwise_fit = auto_arima(df['Prev Close'], trace=True, suppress_warnings=True)
    
    #spliting data into train and test and chossing last 120 days for prediction
    train = price.iloc[:-120]
    test = price.iloc[-120:]

    #applying arima model and printing its summary
    model=SARIMAX(price,order=stepwise_fit.order,seasonal_order=(0,0,0,12))
    model=model.fit()
    print(model.summary())

    #predicting the values using arima for last 120days
    a=len(train)
    b=len(train)+len(test)-1    
    pred=model.predict(start=a,end=b,typ='levels')

    #plotting and saving graph of test vs predictions
    s = name + '_SARIMA_Predictions'
    pred.plot(label= s, legend=True)
    s = name + '_Actual_Data'
    test.plot(label=s, legend=True)
    s = name + '_sarmima_test_Comparision'
    plt.title(s)
    s = name + '_sarmima_test_Comparision.png'
    plt.savefig(s)
    plt.show()

    #plotting and saving graph of whole data and predictions
    s = name + '_Actual_Data'
    price.plot(label=s, legend=True, figsize=(12,5))
    s = name + '_SARIMA_Predictions'
    pred.plot(label= s, legend=True)
    s = name + '_sarmima_all'
    plt.title(s)
    s = name + '_sarmima_all.png'
    plt.savefig(s)
    plt.show()

    #calculating and printing errors between test and pred
    print('Mean Squared Error: ',mean_squared_error(test, pred))
    print('Mean Squared Log Error: ',mean_squared_log_error(test, pred))
    print('Mean Absolute Error: ',mean_absolute_error(test, pred))
    print('Max Error',max_error(test, pred))


# In[3]:


arima_model('Dataset\CIPLA.csv')


# In[4]:


arima_model('Dataset\HDFC.csv')


# In[5]:


arima_model('Dataset\WIPRO.csv')


# In[6]:


arima_model('Dataset\AXISBANK.csv')


# In[11]:


#SARIMA


# In[12]:


sarima_model('Dataset\CIPLA.csv')


# In[13]:


sarima_model('Dataset\HDFC.csv')


# In[14]:


sarima_model('Dataset\WIPRO.csv')


# In[15]:


sarima_model('Dataset\AXISBANK.csv')

