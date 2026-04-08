"""
analysis using SARIMAX
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Use the relative paths for dat for this one.
cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"


all_data = pd.read_csv(data_folder / "all_data_30min.csv")
all_data["datetime"] = pd.to_datetime(all_data["datetime"], yearfirst=True)
all_data.index = all_data["datetime"]

max_data = all_data.groupby([all_data.index.year, all_data.index.month, all_data.index.day]).max()
max_data.index=pd.to_datetime(pd.DataFrame(max_data.index.values.tolist(), columns=['year','month','day']))


max_data.drop("datetime", axis=1, inplace=True)

# normalise the data:
# power demand is logged:
log_demand = np.log1p(max_data["total_demand"])

# rest of data is min-max scaled:
scaler = MinMaxScaler()
scaler.fit(max_data.drop("total_demand", axis=1))
normalized_data = scaler.transform(max_data.drop("total_demand", axis=1))


# repack data:
data = pd.DataFrame.from_dict({"total_demand": log_demand,
                               "rainfall": normalized_data[:, 0],
                               "pv_capacity": normalized_data[:, 1],
                               "temperature": normalized_data[:, 2],
                               "solar_power": normalized_data[:, 3]})

def fourier_series(dates, period, K, t0):
    """
    Generates the fourier series for the seasonality of the ARIMAX model, includes both the Sin and Cosine values.
    :param dates: Dates over which to generate the series
    :param period: Period of the fourier series (in minutes)
    :param K: Order of the fourier series (typically 1-3)
    :param t0: Origin of time series
    :return:
    """

    t = (dates - t0).total_seconds() / 60  # minutes since t0
    freq = 2 * np.pi / period
    X = {}
    for k in range(1, K + 1):
        X[f'sin_{period}_{k}'] = np.sin(freq * k * t)
        X[f'cos_{period}_{k}'] = np.cos(freq * k * t)

    return pd.DataFrame(X, index=dates, columns=list(X.keys()))

start = 0
end_train = 365*7
end_test  = 31

burn_in = 31
data["lag_48"] = data["total_demand"].shift(48)
data["lag_96"] = data["total_demand"].shift(96)

# dump the nan rows
data.dropna(inplace=True, how='any', axis=0)
#weekly_terms = fourier_series(data.index, 365, K=1, t0=data.index[0])


complete_dataset = pd.concat([
    data,
    #                          weekly_terms
                              ], axis=1)
complete_dataset.drop(["pv_capacity", "rainfall", "solar_power"], axis=1, inplace=True)


train_set  = complete_dataset["total_demand"].iloc[start:start+end_train]
test_set = complete_dataset["total_demand"].iloc[start+end_train: start+end_train+end_test]

train_exog = complete_dataset.drop("total_demand", axis=1).iloc[start:start+end_train]

#train_exog = None

test_exog =complete_dataset.drop("total_demand", axis=1).iloc[start+end_train: start+end_train+end_test]

#test_exog = None

model = SARIMAX(endog = train_set,
                exog = train_exog,
                order=(3, 1, 1),
                seasonal_order=(1, 1, 0, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
                )

fit_res1 = model.fit(disp=False,
                     maxiter=500, method='lbfgs')

predict = fit_res1.get_prediction(start=train_set.index[burn_in]).predicted_mean
forecast = fit_res1.get_forecast(steps=end_test, exog=test_exog).predicted_mean

plt.plot(train_set.index, train_set, color='black', label='true')
plt.plot(predict.index, predict, color='grey', label='train')

plt.plot(test_set.index, test_set, color='red')
plt.plot(test_set.index, forecast, color='blue')


print(fit_res1.summary())

plt.show()
