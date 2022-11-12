import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# Importo i dati
df = pd.read_csv("seriefit2021.csv")
x = df.x
y = df.y
plt.plot(x, y, color="blue", label="Fit")
plt.scatter(x, y, color="red")
plt.grid()
plt.title("Dati grezzi")
plt.xlabel("Numero mesi")
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
err2 = mean_squared_error(y_2, p2(x_2))

# Polinomio di grado 3
poly3 = np.polyfit(x_2, y_2, 3)
p3 = np.poly1d(poly3)
err3 = mean_squared_error(y_2, p3(x_2))

# Polinomio di grado 4
poly4 = np.polyfit(x_2, y_2, 4)
p4 = np.poly1d(poly4)
err4 = mean_squared_error(y_2, p4(x_2))

# Polinomio di grado 5
poly5 = np.polyfit(x_2, y_2, 5)
p5 = np.poly1d(poly5)
err5 = mean_squared_error(y_2, p5(x_2))

# Polinomio di grado 6
poly6 = np.polyfit(x_2, y_2, 6)
p6 = np.poly1d(poly6)
err6 = mean_squared_error(y_2, p6(x_2))

# Polinomio di grado 10
poly10 = np.polyfit(x_2, y_2, 10)
p10 = np.poly1d(poly10)
err10 = mean_squared_error(y_2, p10(x_2))

plt.plot(x, y, color="black", label="Fit")
plt.plot(x_2, p2(x_2), color="brown", label="Polinomio Grado 2")
plt.plot(x_2, p3(x_2), color="pink", label="Polinomio Grado 3")
plt.plot(x_2, p4(x_2), color="cyan", label="Polinomio Grado 4")
plt.plot(x_2, p5(x_2), color="olive", label="Polinomio Grado 5")
plt.plot(x_2, p6(x_2), color="grey", label="Polinomio Grado 6")
plt.plot(x_2, p10(x_2), color="orange", label="Polinomio Grado 10")

# Scarto quadratico medio
print("Errore quadratico medio con polinomio di grado 2: ", err2)
print("Errore quadratico medio con polinomio di grado 3: ", err3)
print("Errore quadratico medio con polinomio di grado 4: ", err4)
print("Errore quadratico medio con polinomio di grado 5: ", err5)
print("Errore quadratico medio con polinomio di grado 6: ", err6)
print("Errore quadratico medio con polinomio di grado 6: ", err10)

plt.title("Funzioni di approssimazione")
plt.xlabel("Numero mesi")
plt.ylabel("Valore")
plt.legend()
plt.show()

# Facendo variare la y_2 (quindi anche x_2) per prendere solo una parte dei dati, ho notato che con un polinomio di
# quarto grado lo scarto quadratico è minore, tenendo però in considerazione i vincoli (che in questo caso sono 4)
# Il polinomio di grado 4 sembra potenzialmente quello con approssimazione migliore anche considerando anche meno della
# metà dei dati.
# Ho provato ad approssimare con la distribuzione gaussiana ma senza ottenere risultati soddisfacenti

plt.plot(x, y, color="black", label="Fit")
plt.plot(x_2, p4(x_2), color="cyan", label="Polinomio Grado 4")
plt.title("Funzione di approssimazione")
plt.xlabel("Numero mesi")
plt.ylabel("Valore")
plt.legend()
plt.show()

# Trend, Seasonality e Residuo
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
plt.show()

# Sembra che sia stato già fatto la destagionalizzazione
# Confronto tra stagionalità calcolata attraverso "seasonal_decompose" e quella attraverso calcoli manuali
test = y - p3(x)
plt.plot(x, test, color="red")
plt.plot(x, detrend, color="blue")
plt.show()

# Confronto fra il trend e la funzione di trend
plt.plot(x, result_additive.trend, color="red")
plt.plot(x, p3(x), color="blue")
plt.show()

# Predizione stagionalità
season_coeff = []
for d in [0, 1, 2, 3, 4, 5, 6]:
    season_coeff.append(np.mean(test[d::7]))

final_season = season_coeff[1:]
final_season = np.append(final_season, season_coeff)
final_season = np.resize(final_season, 36)
