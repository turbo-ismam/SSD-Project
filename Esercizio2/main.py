from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import statsmodels.api as sm

# # # # # STATISTIC MODELS # # # # #

# Data Upload
df = pd.read_csv("serieFcast2021.csv")

# Preprocessing
df = df.fillna(df.interpolate())  # filling the NaN values
amount = df["amount"]  # array of amount data
log_data = np.log(amount)  # Log transform
data = pd.Series(log_data)  # Convert to pandas series

df["amount"].plot()
plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Data'])
plt.title('Raw data')
plt.show()

data.plot()
plt.xlabel('time')
plt.legend(['Log data'])
plt.ylabel('log amounts')
plt.title('Log data')

plt.show()

# ACF plot
sm.graphics.tsa.plot_acf(data.values, lags=24)
plt.show()

# train and test set
cutpoint = int(0.7 * len(data))
train = data[:cutpoint]
test = data[cutpoint:]

train_raw = amount[:cutpoint]
test_raw = amount[cutpoint:]
# print(train, test)


# # # AUTOARIMA WITH THE TRAIN DATA # # #

model = pm.auto_arima(train.values, start_p=1, start_q=1,
                      test='adf', max_p=3, max_q=3, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)  # False full grid
print(model.summary())
morder = model.order
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order
print("Sarimax seasonal order {0}".format(mseasorder))

# predictions and forecasts
fitted = model.fit(train)
ypred = fitted.predict_in_sample()  # prediction (in-sample)
yfore = fitted.predict(n_periods=12)  # forecast (out-of-sample)

plt.plot(train, color="green")  # Dati di train
plt.plot(test, color="royalblue")
plt.plot([None for x in range(12)] + [x for x in ypred[12:]], color="orange")
plt.plot([None for x in ypred] + [x for x in yfore], color="red")  # Previsione

plt.xlabel('time')
plt.ylabel('log amounts')
plt.legend(['Train', 'Test', 'Forecast', 'Prediction'])
plt.title('AUTOARIMA with log TRAIN data set')

plt.show()

# recostruction
yplog = pd.Series(ypred)
expdata = np.exp(yplog)  # unlog
expfore = np.exp(yfore)

plt.plot(train_raw, color="green")  # train set grezzo
plt.plot(test_raw, color="royalblue")  # test set grezzo
plt.plot([None for x in range(36)] + [x for x in expdata[36:]], color="orange")
plt.plot([None for x in expdata] + [x for x in expfore], color="red")  # previsione

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Train', 'Test', 'Forecast', 'Prediction'])
plt.title('AUTOARIMA with the reconstructed TRAIN data set')

plt.show()

# # # SARIMAX WITH TRAIN DATA # # #
sarima_model = SARIMAX(train.values, order=morder, seasonal_order=mseasorder)
sfit = sarima_model.fit()
sfit.plot_diagnostics()
plt.show()

ypred = sfit.predict(start=0, end=len(train))
yfore = sfit.get_forecast(steps=36)
expdata = np.exp(ypred)  # unlog
expfore = np.exp(yfore.predicted_mean)

plt.plot(train_raw, color="green")  # raw train data
plt.plot(test_raw, color="royalblue")
plt.plot([None for x in expdata] + [x for x in expfore], color="red")
plt.plot(expdata, color="orange")

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Train', 'Test', 'Prediction', 'Forecast'])
plt.title('SARIMAX with the reconstructed TRAIN data set')

plt.show()

# # # AUTOARIMA WITH FULL DATASET # # #

model = pm.auto_arima(log_data, start_p=1, start_q=1,
                      test='adf', max_p=3, max_q=3, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)  # False full grid
print(model.summary())
morder = model.order
print("Sarimax order {0}".format(morder))
mseasorder = model.seasonal_order
print("Sarimax seasonal order {0}".format(mseasorder))

# predictions and forecasts
fitted = model.fit(log_data)
ypred = fitted.predict_in_sample()  # prediction (in-sample)
yfore = fitted.predict(n_periods=24)  # forecast (out-of-sample)

plt.plot(log_data, color="green")  # Dati di train
plt.plot([None for x in range(24)] + [x for x in ypred[24:]], color="orange")  # Forecast
plt.plot([None for x in ypred] + [x for x in yfore], color="red")  # Predict

plt.xlabel('time')
plt.ylabel('log amounts')
plt.legend(['Train', 'Prediction', 'Forecast'])
plt.title('AUTOARIMA with entire log data set')

plt.show()

# recostruction
yplog = pd.Series(ypred)
expdata = np.exp(yplog)  # unlog
expfore = np.exp(yfore)

plt.plot(amount, color="green")  # train set grezzo
plt.plot([None for x in expdata] + [x for x in expfore], color="red")  # previsione
plt.plot([None for x in range(24)] + [x for x in expdata[24:]], color="orange")

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Train', 'Prediction', 'Forecast'])
plt.title('AUTOARIMA with entire reconstructed data set')

plt.show()

# # # SARIMAX WITH FULL DATASET # # #

sarima_model = SARIMAX(log_data, order=morder, seasonal_order=mseasorder)
sfit = sarima_model.fit()
sfit.plot_diagnostics()
plt.show()

ypred = sfit.predict(start=0, end=len(log_data))
yfore = sfit.get_forecast(steps=24)
expdata = np.exp(ypred)  # unlog
expfore = np.exp(yfore.predicted_mean)

plt.plot(expdata, color="orange")
plt.plot(amount, color="green")
plt.plot([None for x in expdata] + [x for x in expfore], color="red")

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Forecast', 'Raw train data', 'Prediction'])
plt.title('SARIMAX with entire reconstructed data set')

plt.show()
