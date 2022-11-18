from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential  # pip install keras
from keras.layers import Dense  # pip install tensorflow (as administrator)
import os, math
import pmdarima as pm  # pip install pmdarima
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm

df = pd.read_csv("serieFcast2021.csv")

ds = df.Series("y")
print(ds)

print(df)
df["amount"].plot()
plt.show()
