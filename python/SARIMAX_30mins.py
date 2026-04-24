"""
analysis using SARIMAX
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from public_holidays import get_holidays
import scipy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pmdarima as pm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Use the relative paths for dat for this one.
cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"


all_data = pd.read_csv(data_folder / "all_data_30min.csv")
all_data["datetime"] = pd.to_datetime(all_data["datetime"], yearfirst=True)
all_data.index = all_data["datetime"]
all_data.drop("datetime", axis=1, inplace=True)

all_data["T2"] = all_data["temperature"]**2

# remove the power guard values, fix the data import.
all_data = all_data.iloc[48:, :]

# normalise the data:
# power demand is logged:
log_demand = np.log1p(all_data["total_demand"])

# rest of data is min-max scaled:
scaler = MinMaxScaler()
scaler.fit(all_data.drop("total_demand", axis=1))
normalized_data = scaler.transform(all_data.drop("total_demand", axis=1))

holidays = []
for i in range(10):
    year = 2010+i
    holidays.append(get_holidays(year))

holidays = [dt for sublist in holidays for dt in sublist]


one_hot_holidays = np.zeros_like(log_demand.index, dtype=int)
one_hot_weekdays = np.zeros_like(log_demand.index, dtype=int)

for enum, day_of_index in enumerate(log_demand.index):

    temp_date = datetime.date(day_of_index.year, day_of_index.month, day_of_index.day)

    if temp_date in holidays:
        one_hot_holidays[enum] = 1

    if temp_date.weekday() == 5 or temp_date.weekday() == 6:
        one_hot_weekdays[enum] = 1

# repack data:
data = pd.DataFrame.from_dict({"total_demand": log_demand,
                               "rainfall": normalized_data[:, 0],
                               "holidays": one_hot_holidays,
                               "weekends": one_hot_weekdays,
                               "pv_capacity": normalized_data[:, 1],
                               "temperature": normalized_data[:, 2],
                               "solar_power": normalized_data[:, 3],
                               "t2": normalized_data[:, 4]})

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


aic = []

plots = False

all_in_sample = []
all_out_sample = []
SARIMAX_modes = []
daily_max = []

jump = 4

fourier_yearly = fourier_series(data.index, 365.25, K=1, t0=data.index[0])

burn_in = 48 * 2
data["temp_1_sin"] = data["temperature"].shift(1).values * fourier_yearly.iloc[:, 0].values
data["temp_9_sin"] = data["temperature"].shift(9).values * fourier_yearly.iloc[:, 0].values
data["solar_4_sin"] = data["solar_power"].shift(4).values * fourier_yearly.iloc[:, 0].values
data["solar_16_sin"] = data["solar_power"].shift(16).values * fourier_yearly.iloc[:, 0].values

data["temp_1_cos"] = data["temperature"].shift(1).values * fourier_yearly.iloc[:, 1].values
data["temp_9_cos"] = data["temperature"].shift(9).values * fourier_yearly.iloc[:, 1].values
data["solar_4_cos"] = data["solar_power"].shift(4).values * fourier_yearly.iloc[:, 1].values
data["solar_16_cos"] = data["solar_power"].shift(16).values * fourier_yearly.iloc[:, 1].values

data["day"] = data.index.day

data["lag_48*7"] = data["total_demand"].shift(48 * 7)

# dump the nan rows
data.dropna(inplace=True, how='any', axis=0)
weekly_terms = fourier_series(data.index, 7 * 48, K=1, t0=data.index[0])

# just to remove an annoying warning :)
data.index = pd.DatetimeIndex(data.index, freq='30min')

complete_dataset = pd.concat([data, weekly_terms], axis=1)
complete_dataset.drop([
    #"pv_capacity",
    "rainfall",
    "holidays",
    #"t2",
    "weekends",
], axis=1, inplace=True)

params = None

for steps in range(1):

    print(steps)

    start = 48*200 + steps*48*jump
    end_train = 48*28
    end_test  = 48*1


    train_set  = complete_dataset["total_demand"].iloc[start:start+end_train]
    test_set = complete_dataset["total_demand"].iloc[start+end_train: start+end_train+end_test]

    train_exog = complete_dataset.drop("total_demand", axis=1).iloc[start:start+end_train]
    test_exog =complete_dataset.drop("total_demand", axis=1).iloc[start+end_train: start+end_train+end_test]

    model = SARIMAX(endog=train_set,
                                    exog=train_exog,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 0, 1, 48),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    fit_res1 = model.fit(disp=True, maxiter=1000, start_params=None)

    params = fit_res1.params


    predict = fit_res1.get_prediction(start=train_set.index[burn_in]).predicted_mean
    forecast = fit_res1.get_forecast(steps=test_exog.shape[0], exog=test_exog).predicted_mean

    daily_max.append((test_set.index.max().date(), forecast.max(), test_set.max()))

    in_sample_values = (mean_squared_error(train_set[burn_in:], predict),
                                        mean_absolute_error(train_set[burn_in:], predict),
                                        r2_score(train_set[burn_in:], predict),
                                        mean_absolute_percentage_error(train_set[burn_in:], predict))

    out_sample_values = (mean_squared_error(test_set, forecast),
                         mean_absolute_error(test_set, forecast),
                         r2_score(test_set, forecast),
                         mean_absolute_percentage_error(test_set, forecast))

    all_in_sample.append(in_sample_values)
    all_out_sample.append(out_sample_values)


    if plots:
        plt.plot(train_set.index[burn_in:], train_set[burn_in:], color='black', label='true')
        plt.plot(predict.index, predict, color='grey', label='train')

        plt.plot(test_set.index, test_set, color='red')
        plt.plot(test_set.index, forecast, color='blue')

        plt.show()

    #print(fit_res1.summary())


results = pd.DataFrame(daily_max, columns=["date", "prediction", "actual"])
autosolve = pm.auto_arima(y=train_set, X=train_exog,m=48, trace=True)

if False:
    for i in range(1, 6):
        for j in range(1, 6):
            for k in range(1, 6):
                print(f"Testing: {i} {j} {k}")

                model = SARIMAX(endog=train_set,
                                exog=train_exog,
                                order=(i, j, k),
                                seasonal_order=(1, 1, 1, 48),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                                )

                fit_res1 = model.fit(disp=False,
                                     maxiter=500, method='lbfgs')

                predict = fit_res1.get_prediction(start=train_set.index[burn_in]).predicted_mean
                forecast = fit_res1.get_forecast(steps=test_exog.shape[0], exog=test_exog).predicted_mean

                in_sample_values = (mean_squared_error(train_set[burn_in:], predict),
                                    mean_absolute_error(train_set[burn_in:], predict),
                                    r2_score(train_set[burn_in:], predict),
                                    mean_absolute_percentage_error(train_set[burn_in:], predict))

                out_sample_values = (mean_squared_error(test_set, forecast),
                                     mean_absolute_error(test_set, forecast),
                                     r2_score(test_set, forecast),
                                     mean_absolute_percentage_error(test_set, forecast))

                SARIMAX_modes.append((i, j, k))
                all_in_sample.append(in_sample_values)
                all_out_sample.append(out_sample_values)
