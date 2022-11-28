import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import math

# Importo i dati
df = pd.read_csv("seriefit2021.csv")
x = df.x
y = df.y

plt.plot(x, y, color="blue", label="Fit")
plt.scatter(x, y, color="red")
plt.title("Dati grezzi")
plt.xlabel("Mesi")
plt.ylabel("Valore")
plt.legend()
plt.show()

# Funzione di trend
# Creo delle copie dei valori dei miei assi, serviranno per prendere solo una parte dei valori
y_2 = y[:len(x)]
x_2 = x[0:len(y)]

# Approssimo con un polinomio di grado 2
poly2 = np.polyfit(x_2, y_2, 2)
p2 = np.poly1d(poly2)
mse_poly2 = mean_squared_error(y_2, p2(x_2))
rmse_poly2 = math.sqrt(mse_poly2)

# Polinomio di grado 3
poly3 = np.polyfit(x_2, y_2, 3)
p3 = np.poly1d(poly3)
mse_poly3 = mean_squared_error(y_2, p3(x_2))
rmse_poly3 = math.sqrt(mse_poly3)

# Polinomio di grado 4
poly4 = np.polyfit(x_2, y_2, 4)
p4 = np.poly1d(poly4)
mse_poly4 = mean_squared_error(y_2, p4(x_2))
rmse_poly4 = math.sqrt(mse_poly4)

# Polinomio di grado 5
poly5 = np.polyfit(x_2, y_2, 5)
p5 = np.poly1d(poly5)
mse_poly5 = mean_squared_error(y_2, p5(x_2))
rmse_poly5 = math.sqrt(mse_poly5)

# Polinomio di grado 6
poly6 = np.polyfit(x_2, y_2, 6)
p6 = np.poly1d(poly6)
mse_poly6 = mean_squared_error(y_2, p6(x_2))
rmse_poly6 = math.sqrt(mse_poly6)

# Polinomio di grado 10
poly10 = np.polyfit(x_2, y_2, 10)
p10 = np.poly1d(poly10)
mse_poly10 = mean_squared_error(y_2, p10(x_2))
rmse_poly10 = math.sqrt(mse_poly10)

plt.plot(x, y, color="black", label="Fit")
plt.plot(x_2, p2(x_2), color="brown", label="Polinomio Grado 2")
plt.plot(x_2, p3(x_2), color="pink", label="Polinomio Grado 3")
plt.plot(x_2, p4(x_2), color="cyan", label="Polinomio Grado 4")
plt.plot(x_2, p5(x_2), color="olive", label="Polinomio Grado 5")
plt.plot(x_2, p6(x_2), color="grey", label="Polinomio Grado 6")
plt.plot(x_2, p10(x_2), color="orange", label="Polinomio Grado 10")

# MSE & RMSE dei polinomi
print('Polinomio grado 2 | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_poly2, rmse_poly2))
print('Polinomio grado 3 | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_poly3, rmse_poly3))
print('Polinomio grado 4 | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_poly4, rmse_poly4))
print('Polinomio grado 5 | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_poly5, rmse_poly5))
print('Polinomio grado 6 | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_poly6, rmse_poly6))
print('Polinomio grado 10 | MSE: {0:0.3f} | RMSE: ({1:0.3f})'.format(mse_poly10, rmse_poly10))

plt.title("Funzioni di approssimazione")
plt.xlabel("Numero mesi")
plt.ylabel("Valore")
plt.legend()
plt.show()

# Osservando gli andamenti delle funzioni e i relativi MSE ritengo che i polinomi di grado 2 e 4 siano le migliori
# approssimazioni possibili.

plt.plot(x, y, color="grey", label="Fit")
plt.plot(x_2, p2(x_2), color="green", label="Polinomio Grado 2")
plt.plot(x_2, p4(x_2), color="orange", label="Polinomio Grado 4")

plt.title("Funzioni di approssimazione")
plt.xlabel("Numero mesi")
plt.ylabel("Valore")
plt.legend()
plt.show()

# Trend, Seasonality e Residuo con modello additivo
ds = df[df.columns[1]]
result_additive = seasonal_decompose(ds, model='additive', period=12)
result_additive.plot()
plt.show()

# Detrend
detrend = y - result_additive.trend
plt.plot(x, result_additive.trend, color="red", label="Trend")
plt.plot(x, detrend, color="green", label="Detrend")
plt.title("Detrend")
plt.legend()
plt.show()

result_multiplicative = seasonal_decompose(ds, model='multiplicative', period=12)
result_multiplicative.plot()
plt.show()

# Un modello additivo risulta migliore di quello moltiplicativo per eliminare l'effetto del trend
residuo = result_additive.resid
stagione = result_additive.seasonal

# ACF
diff_data = ds.diff()
# reset 1st elem
diff_data[0] = ds[0]
sm.graphics.tsa.plot_acf(diff_data, lags=36)
plt.show()

# residuo e stagionalità
destagionalizzato = detrend - result_additive.seasonal
plt.plot(x, destagionalizzato, color="red")
plt.plot(x, result_additive.seasonal, color="green")
plt.scatter(x, result_additive.seasonal, color='orange')

plt.title("Residuo e Stagionalità")
plt.legend(["Destagionalizzato", "Stagionalità"])

plt.show()

# Sembra che sia stato già fatto la destagionalizzazione
# Confronto tra stagionalità calcolata attraverso "seasonal_decompose" e quella attraverso calcoli manuali
test = y - p4(x)
plt.plot(x, test)
plt.plot(x, detrend)

plt.title("Detrend - calcolato vs ottenuto")
plt.legend(["Calcolato", "Ottenuto"])

plt.show()

# Confronto fra il trend e la funzione di trend
plt.plot(x, result_additive.trend)
plt.plot(x, p2(x))
plt.plot(x, p4(x))

plt.title("Trend e funzioni di trend")
plt.legend(["Trend", "Polinomio grado 2", "Polinomio grado 4"])

plt.show()

# Predizione stagionalità
season_coeff = []
for d in [0, 1, 2, 3, 4, 5, 6]:
    season_coeff.append(np.mean(test[d::7]))

final_season = season_coeff[1:]
final_season = np.append(final_season, season_coeff)
final_season = np.resize(final_season, 36)

# Predizione per ulteriori 36 periodi di tempo

x1 = np.linspace(65, 101, 36)
x2 = np.linspace(65, 101, 36)
predictions_poly2 = (p2(x1)) + final_season
predictions_poly4 = (p4(x2)) + final_season

plt.plot(x, y)
plt.plot(x1, predictions_poly2)
plt.plot(x2, predictions_poly4)

plt.show()
