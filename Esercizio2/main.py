import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pmdarima as pm
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("serieFcast2021.csv")
ds = df[df.columns[1]]
plt.plot(df.amount, label="Fcast")
plt.grid()
plt.legend()
plt.show()

# Preprocessing
# adding missing values
ds1 = ds.interpolate()
plt.plot(ds1, label="Fcast")
plt.grid()
plt.show()
x = df.iloc[:, 0]
y = ds1

npa = ds1.to_numpy()
logdata = np.log(npa)
plt.plot(npa, color='blue', marker="o")
plt.plot(logdata, color='red', marker="o")
plt.title("numpy.log()")
plt.xlabel("x");
plt.ylabel("logdata")
plt.show()

# Logdiff
logdiff = pd.Series(logdata).diff()
cutpoint = int(0.7 * len(logdiff))
trainLog = logdiff[:cutpoint]
testLog = logdiff[cutpoint:]
trainLog[0] = 0  # set first entry
reconstruct = np.exp(np.r_[trainLog, testLog].cumsum() + logdata[0])

# Seasonal decompose
result1 = seasonal_decompose(ds1, model='multiplicative', period=12)
result1.plot()
plt.show()

# ACF
diffdata = ds1.diff()
diffdata[0] = ds1[0]  # reset 1st elem
acfdata = acf(diffdata, nlags=24)
plt.bar(np.arange(len(acfdata)), acfdata)
plt.show()
# otherwise
sm.graphics.tsa.plot_acf(diffdata, lags=24)
plt.show()

# STATISTICAL METHODS

# AUTOARIMA
model = pm.auto_arima(ds1.values, start_p=1, start_q=1,
                      test='adf', max_p=3, max_q=3, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)  # False full grid
print(model.summary())

morder = model.order
mseasorder = model.seasonal_order
fitted = model.fit(ds1)
yfore = fitted.predict(n_periods=24)  # forecast
ypred = fitted.predict_in_sample()
plt.plot(ds1.values)
plt.plot(ypred)
plt.plot([None for i in ypred] + [x for x in yfore])
plt.xlabel('time');
plt.ylabel('amount')
plt.show()

# Prova con Sarima sui dati in log
sarima_model = SARIMAX(trainLog, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12))
sfit = sarima_model.fit()
sfit.plot_diagnostics(figsize=(10, 6))
plt.show()
ypred = sfit.predict(start=0, end=len(ds1) - 1)
print("MSE={}".format(mean_absolute_error(testLog, ypred[cutpoint:])))

forewrap = sfit.get_forecast(steps=24)
forecast_ci = forewrap.conf_int()
forecast_val = forewrap.predicted_mean
plt.plot(logdiff)
plt.plot(ypred)
plt.plot(np.linspace(len(logdiff), len(logdiff) + 24, 24), forecast_val)
plt.xlabel('time');
plt.ylabel('amount')
plt.show()

# Prova con Sarima sui dati ORIGINALI
sarima_model = SARIMAX(ds1, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12))
sfit = sarima_model.fit()
sfit.plot_diagnostics(figsize=(10, 6))
plt.show()
ypred = sfit.predict(start=0, end=len(ds1) - 1)
print("MSE={}".format(mean_absolute_error(ds1, ypred)))

forewrap = sfit.get_forecast(steps=24)
forecast_ci = forewrap.conf_int()
forecast_val = forewrap.predicted_mean
plt.plot(ds1)
plt.plot(ypred)
plt.plot(np.linspace(len(ds1), len(ds1) + 24, 24), forecast_val)
plt.xlabel('time')
plt.ylabel('amount')
plt.show()

# Sembra sia meglio con il log (dal mse)
