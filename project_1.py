

import pandas as pd
import numpy as np
import pandas
from sklearn.svm import LinearSVR
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.svm import SVR
# plt.style.use('fivethirtyeight')
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression


def stock_predict():
	import pandas as pd
	df=pandas.read_csv('Dataset/WIPRO.csv')

	df=df.tail(22)

	X = pandas.read_csv('Dataset/WIPRO.csv', header=0)
	X=X.tail(22)

	y = pandas.read_csv('Dataset/WIPRO.csv', header=0)
	y=y.tail(22)


	days=[]
	prev_close=[]
	df_date=df.loc[:,'Date']   ########### GET the Date
	df_prev_close=df.loc[:,'Prev Close']  ###### GET the Close Price 

	for d in df_date:
	  days.append([int (d.split('-')[2])])  ###########3 for predicting

	for p in df_prev_close:
	  prev_close.append(float(p))


	#SVR MODELS

	#traing model ::::::::::::::: Linear
	svr1=SVR(kernel='linear',C=1000.0)
	svr1=svr1.fit(days,prev_close)

	print("Linear Done")

	#traing model ::::::::::::::: Plolynomila
	svr2=SVR(kernel='poly',C=1000.0, degree=2)
	svr2.fit(days,prev_close)

	print("Poly Done")

	#traing model ::::::::::::::: Plolynomila
	svr3=SVR(kernel='rbf',C=1000.0)
	svr3.fit(days,prev_close)

	print("RBF Done")
	import matplotlib.pyplot as plt

	plt.figure(figsize=(16,8))
	plt.xticks(rotation=45)
	plt.ylabel('Prev Closing Price')
	plt.xlabel('Date')
	plt.plot(df['Date'],prev_close,color='red',label='from data')
	plt.plot(df['Date'],svr1.predict(days),color='orange',label='Linear')
	plt.plot(df['Date'],svr2.predict(days),color='pink',label='Polynomial')
	plt.plot(df['Date'],svr3.predict(days),color='green',label='RBF')
	plt.legend()
	plt.show()






	print("Linear Score = ",svr1.score(days,prev_close))
	print("Ploynomial Score = ",svr2.score(days,prev_close))
	print("RBF Score = ",svr3.score(days,prev_close))
	print(svr1.predict([[22]]))



	############################################## NEXT COMPANY
	df=pd.read_csv('Dataset/AXISBANK.csv')
	df.shape

	df=df.tail(22)

	X = pd.read_csv('Dataset/AXISBANK.csv', header=0)
	X=X.tail(22)

	y = pd.read_csv('Dataset/AXISBANK.csv', header=0)
	y=y.tail(22)



	days=[]
	prev_close=[]
	df_date=df.loc[:,'Date']   ########### GET the Date
	df_prev_close=df.loc[:,'Prev Close']  ###### GET the Close Price 

	for d in df_date:
	  days.append([int (d.split('-')[2])])  ###########3 for predicting

	for p in df_prev_close:
	  prev_close.append(float(p))


	#SVR MODELS

	#traing model ::::::::::::::: Linear
	svr1=SVR(kernel='linear',C=1000.0)
	svr1=svr1.fit(days,prev_close)

	print("Linear Done")

	#traing model ::::::::::::::: Plolynomila
	svr2=SVR(kernel='poly',C=1000.0, degree=2)
	svr2.fit(days,prev_close)

	print("Poly Done")

	#traing model ::::::::::::::: Plolynomila
	svr3=SVR(kernel='rbf',C=1000.0)
	svr3.fit(days,prev_close)

	print("RBF Done")


	plt.figure(figsize=(16,8))
	plt.xticks(rotation=45)
	plt.ylabel('Prev Closing Price')
	plt.xlabel('Date')
	plt.plot(df['Date'],prev_close,color='red',label='from data')
	plt.plot(df['Date'],svr1.predict(days),color='orange',label='Linear')
	plt.plot(df['Date'],svr2.predict(days),color='pink',label='Polynomial')
	plt.plot(df['Date'],svr3.predict(days),color='green',label='RBF')
	plt.legend()
	plt.show()


	print("Linear Score = ",svr1.score(days,prev_close))
	print("Ploynomial Score = ",svr2.score(days,prev_close))
	print("RBF Score = ",svr3.score(days,prev_close))












	#############################################   ARIMA 
	X = pd.read_csv('Dataset/WIPRO.csv', header=0)
	da=X['Date']
	X=X.drop('Date',axis=1)
	X=X.drop('Symbol',axis=1)
	X=X.drop('Series',axis=1)
	X=X.drop('Prev Close',axis=1)
	X=X.drop('Turnover',axis=1)
	X=X.drop('Trades',axis=1)
	X=X.drop('Deliverable Volume',axis=1)
	X=X.drop('%Deliverble',axis=1)
	X=X.iloc[500:]
	da=da.iloc[500:]
	# print(X.head(100))


	"""**PART 2**"""

	################ NEX DAY PREDICTION

	df=pd.read_csv('Dataset/WIPRO.csv')
	df.shape


	df=df.tail(100)

	X = pd.read_csv('Dataset/WIPRO.csv', header=0)
	X=X.tail(22)

	y = pd.read_csv('Dataset/WIPRO.csv', header=0)
	y=y.tail(100)

	# print(y)

	days=[]
	prev_close=[]
	df_date=df.loc[:,'Date']   ########### GET the Date
	df_prev_close=df.loc[:,'Prev Close']  ###### GET the Close Price 

	for d in df_date:
	  days.append([int (d.split('-')[2])])  ###########3 for predicting

	for p in df_prev_close:
	  prev_close.append(float(p))

	# y=y.reshape(-1,1)
	print(days)
	print(prev_close)


	df1 = pd.DataFrame(prev_close,columns=['value'])
	df1

	import numpy as np, pandas as pd
	from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
	import matplotlib.pyplot as plt
	plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

	# Original Series
	fig, axes = plt.subplots(3, 2, sharex=True)
	axes[0, 0].plot(df1.value); axes[0, 0].set_title('Original Series')
	plot_acf(df1.value, ax=axes[0, 1])

	# 1st Differencing
	axes[1, 0].plot(df1.value.diff()); axes[1, 0].set_title('1st Order Differencing')
	plot_acf(df1.value.diff().dropna(), ax=axes[1, 1])

	# 2nd Differencing
	axes[2, 0].plot(df1.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
	plot_acf(df1.value.diff().diff().dropna(), ax=axes[2, 1])

	plt.show()


	fig, axes = plt.subplots(1, 2, sharex=True)
	axes[0].plot(df1.value.diff()); axes[0].set_title('1st Differencing')
	axes[1].set(ylim=(0,5))
	plot_pacf(df1.value.diff().dropna(), ax=axes[1])

	plt.show()

	#####################   ORDER OF MA TERM
	fig, axes = plt.subplots(1, 2, sharex=True)
	axes[0].plot(df1.value.diff()); axes[0].set_title('1st Differencing')
	axes[1].set(ylim=(0,1.2))
	plot_acf(df1.value.diff().dropna(), ax=axes[1])

	plt.show()


	# 1,1,2 ARIMA Model
	model = ARIMA(df1.value, order=(2,0,2))
	model_fit = model.fit()
	print(model_fit.summary())


	# 1,1,1 ARIMA Model
	model = ARIMA(df1.value, order=(1,1,1))
	model_fit = model.fit()
	print(model_fit.summary())

	residuals = pd.DataFrame(model_fit.resid)
	fig, ax = plt.subplots(1,2)
	residuals.plot(title="Residuals", ax=ax[0])
	residuals.plot(kind='kde', title='Density', ax=ax[1])
	plt.show()

	# Actual vs Fitted
	model_fit.plot_predict(dynamic=False)
	plt.show()

	train = df1.value[:85]
	test = df1.value[85:]

	# Build Model
	# model = ARIMA(train, order=(3,2,1))  
	model = ARIMA(train, order=(1, 1, 1))  
	fitted = model.fit(disp=-1)  

	# Forecast
	fc, se, conf = fitted.forecast(15, alpha=0.05)  # 95% conf

	# Make as pandas series
	fc_series = pd.Series(fc, index=test.index)
	lower_series = pd.Series(conf[:, 0], index=test.index)
	upper_series = pd.Series(conf[:, 1], index=test.index)

	# Plot
	plt.figure(figsize=(12,5), dpi=100)
	plt.plot(train, label='training')
	plt.plot(test, label='actual')
	plt.plot(fc_series, label='forecast')
	plt.fill_between(lower_series.index, lower_series, upper_series, 
	                 color='k', alpha=.15)
	plt.title('Forecast vs Actuals')
	plt.legend(loc='upper left', fontsize=8)
	plt.show()

	# Build Model
	model = ARIMA(train, order=(3, 2, 0))  
	fitted = model.fit(disp=-1)  
	print(fitted.summary())

	# Forecast
	fc, se, conf = fitted.forecast(15, alpha=0.05)  # 95% conf

	# Make as pandas series
	fc_series = pd.Series(fc, index=test.index)
	lower_series = pd.Series(conf[:, 0], index=test.index)
	upper_series = pd.Series(conf[:, 1], index=test.index)

	# Plot
	plt.figure(figsize=(12,5), dpi=100)
	plt.plot(train, label='training')
	plt.plot(test, label='actual')
	plt.plot(fc_series, label='forecast')
	plt.fill_between(lower_series.index, lower_series, upper_series, 
	                 color='k', alpha=.15)
	plt.title('Forecast vs Actuals')
	plt.legend(loc='upper left', fontsize=8)
	plt.show()

	# Accuracy metrics
	from statsmodels.tsa.stattools import acf
	def forecast_accuracy(forecast, actual):
	    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
	    me = np.mean(forecast - actual)             # ME
	    mae = np.mean(np.abs(forecast - actual))    # MAE
	    mpe = np.mean((forecast - actual)/actual)   # MPE
	    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
	    corr = np.corrcoef(forecast, actual)[0,1]   # corr
	    mins = np.amin(np.hstack([forecast[:,None], 
	                              actual[:,None]]), axis=1)
	    maxs = np.amax(np.hstack([forecast[:,None], 
	                              actual[:,None]]), axis=1)
	    minmax = 1 - np.mean(mins/maxs)             # minmax
	    acf1 = acf(fc-test)[1]                      # ACF1
	    return({'mape':mape, 'me':me, 'mae': mae, 
	            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
	            'corr':corr, 'minmax':minmax})

	forecast_accuracy(fc, test.values)  #############  96.6 from MAPE



if __name__ == "__main__":
    print("PhD19006")
    stock_predict()