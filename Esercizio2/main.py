from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential  # pip install keras
from keras.layers import Dense  # pip install tensorflow (as administrator)
import math
from sklearn.metrics import mean_squared_error
from collections import deque

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
plt.plot([None for x in ypred] + [x for x in yfore], color="red")  # forecast
plt.plot([None for x in range(12)] + [x for x in ypred[12:]], color="orange")  # prediction

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

# MSE & RMSE on train set
mse_train = mean_squared_error(expdata[1:], train_raw)
rmse_train = math.sqrt(mse_train)
print('Train Score - SARIMAX | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_train, rmse_train))

# MSE & RMSE on test set
print(len([None for x in expdata] + [x for x in expfore]), len(test_raw))
tmp_none = [None for x in expdata] + [x for x in expfore]
tmp_clear = []
for elem in tmp_none:
    if elem != None:
        tmp_clear.append(elem)
print(len(tmp_clear), len(test_raw[:36]))
mse_test = mean_squared_error(tmp_clear, test_raw[:36])
rmse_test = math.sqrt(mse_test)
print('Test Score - SARIMAX | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_test, rmse_test))

plt.plot(train_raw, color="green")  # raw train data
plt.plot(test_raw, color="royalblue")
plt.plot(expdata, color="orange")
plt.plot([None for x in expdata] + [x for x in expfore], color="red")

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
plt.plot([None for x in ypred] + [x for x in yfore], color="red")  # Predict
plt.plot([None for x in range(24)] + [x for x in ypred[24:]], color="orange")  # Forecast

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
plt.plot([None for x in range(24)] + [x for x in expdata[24:]], color="orange")
plt.plot([None for x in expdata] + [x for x in expfore], color="red")  # forecast

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

# MSE & RMSE on full data set
mse_train = mean_squared_error(expdata[1:], amount)
rmse_train = math.sqrt(mse_train)
print('Full dataset score - SARIMAX | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_train, rmse_train))

plt.plot([None for x in expdata] + [x for x in expfore], color="red")
plt.plot(amount, color="green")
plt.plot(expdata, color="orange")

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Forecast', 'Raw train data', 'Prediction'])
plt.title('SARIMAX with entire reconstructed data set')

plt.show()

# # # # # MACHINE LEARNING METHODS # # # # #

periods = 12
forcast = 24

ds = pd.read_csv('serieFcast2021.csv')
ds = ds[['amount']]
ds = ds.interpolate()
logged = np.log(ds)

# ds.plot()
# plt.show()
# logged.plot()
# plt.show()

mod_1 = logged.copy(deep=True)

# Aggiungo 12 colonne dove ognuna è il diff della riga n periodi prima
for i in range(periods, 0, -1):
    mod_1[f'amount_{i}'] = mod_1['amount'].diff(periods=i)
# elimino le prime 12 righe perchè contengono i NaN generati dall'operazione precedente
# print(mod_1)
mod_1 = mod_1.dropna()
# print(mod_1)

x = mod_1.drop(columns=['amount']).values
y = mod_1['amount'].values

model = RandomForestRegressor().fit(x, y)

forcasts = []
coda = deque(x[-1], maxlen=periods)

# Uso l'ultima riga per fare la previsione successiva
# Itero questa procedura per 24 volte e ogni volta rimuovo l'elemento più vecchio e metto la mia nuova y
for _ in range(forcast):
    y = model.predict([coda])[0]
    forcasts.append(y)
    coda.append(y)

# print(forcasts)
# print(coda)

reversed = np.exp(forcasts)
indexes = range(ds.index[-1] + 1, ds.index[-1] + forcast + 1)
forcast_series = pd.Series(reversed, index=indexes)

predict = np.exp(model.predict(x))

ds['amount'].plot()
forcast_series.plot()
plt.plot(predict)


plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Dataset', 'Forecast','Predict'])
plt.title('Random Forest Regressor with full dataset')

plt.show()

# MSE & RMSE
raw_data = ds["amount"]
raw_data = raw_data[:240]

mse = mean_absolute_error(raw_data.values, predict)
rmse = math.sqrt(mse)
print('Full dataset score | Random Forest Regressor | MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(mse, rmse))


# # # # # PREDICTIVE NEURAL METHODS # # # # #

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# for reproducibility
np.random.seed(550)

# Importing again the dataset
df = pd.read_csv("serieFcast2021.csv", usecols=[1], names=["amount"], header=0).interpolate()

# time series values
dataset = df.values
# needed for MLP input
dataset = dataset.astype("float32")

# testing with a different split size from the previous one
train_size = int(len(dataset) - 12)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Len train={0}, len test={1}".format(len(train), len(test)))

# sliding window matrices (look_back = window width); dim = n - look_back - 1
look_back = 2
testdata = np.concatenate((train[-look_back:], test))
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(testdata, look_back)

# # # MULTILAYER PERCEPTRON MODEL # # #

loss_function = 'mean_squared_error'
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))  # 8 hidden neurons
model.add(Dense(1))  # 1 output neuron
model.compile(loss=loss_function, optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score - MLP train | MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score - MLP test | MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))
# generate predictions for training and forecast for plotting
trainPredict = model.predict(trainX)
testForecast = model.predict(testX)

# predict for 24 more periods
result = testForecast
for n in np.linspace(0, 23, 24):
    np.append(result, model.predict(np.asarray([[result[-2:, 0][0], result[-1:, 0][0]]],
                                               dtype="float32")))

plt.plot(dataset)
plt.plot(np.concatenate((np.full(look_back - 1, np.nan), trainPredict[:, 0])))
plt.plot(np.concatenate((np.full(len(train) - 1, np.nan), testForecast[:, 0])))
plt.plot(np.concatenate((np.full(len(train) + 11, np.nan), result[:, 0])))

plt.xlabel('time')
plt.ylabel('amounts')
plt.legend(['Dataset', 'Train Predict', 'Test Forecast', 'Prediction'])
plt.title('Multilayer Perceptron Model')

plt.show()

mse_mlp_full = mean_absolute_error(testY, testForecast)
rmse_mlp_full = math.sqrt(mse_mlp_full)

print('Full dataset score | MLP | MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(mse_mlp_full, rmse_mlp_full))
