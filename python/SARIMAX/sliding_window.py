"""
Sliding window for the SARIMAX model using the previously determined parameter values
"""
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from python.public_holidays import get_holidays


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


if __name__=="__main__":

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = pd.read_csv(data_folder / "all_data_30min.csv")
    data["datetime"] = pd.to_datetime(data["datetime"], yearfirst=True)
    data.index = data["datetime"]
    data.drop("datetime", axis=1, inplace=True)

    # remove the power guard values, fix the data import.
    data = data.iloc[48:, :]

    # power demand is logged:
    log_demand = np.log1p(data["total_demand"])

    # rest of data is min-max scaled:
    scaler = MinMaxScaler()
    scaler.fit(data.drop("total_demand", axis=1))
    normalized_data = scaler.transform(data.drop("total_demand", axis=1))

    holidays = []
    for i in range(10):
        year = 2010 + i
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
                                   })
    weekly_terms = fourier_series(data.index, 7 * 48, K=1, t0=data.index[0])

    data["temp_1"] = data["temperature"].shift(1).values
    data["temp_9"] = data["temperature"].shift(9).values
    data["solar_4"] = data["solar_power"].shift(4).values
    data["solar_16"] = data["solar_power"].shift(16).values
    data["lag_48*7"] = data["total_demand"].shift(48 * 7)
    data = pd.concat([data, weekly_terms], axis=1)

    start = datetime.datetime(year=2020, month=1, day=1)
    end = start + datetime.timedelta(days=7 * 8)

    train_set = data[start:end]["total_demand"]
    train_exog = data[start:end].drop("total_demand", axis=1)

    test_set = data[end:end+datetime.timedelta(days=1)]["total_demand"]
    test_exog = data[end:end+datetime.timedelta(days=1)].drop("total_demand", axis=1)


    model = SARIMAX(endog=train_set, exog=train_exog ,order=(2,1,0), seasonal_order=(2,1,0,48), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=-1)

    prediction = model_fit.get_forecast(steps=test_exog.shape[0], exog=test_exog).predicted_mean

    mean_absolute_percentage_error(test_set, prediction)

    model2 = SARIMAX(endog=train_set, order=(2, 1, 0), seasonal_order=(2, 1, 0, 48),
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit2 = model2.fit(disp=-1)
    prediction2 = model_fit2.get_forecast(steps=test_exog.shape[0]).predicted_mean
    mean_absolute_percentage_error(test_set, prediction2)

    test_exog2 = data[start:end].drop(["total_demand", "holidays"], axis=1)
    model3 = SARIMAX(endog=train_set, exog=test_exog2, order=(2, 1, 0), seasonal_order=(2, 1, 0, 48),
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit3 = model.fit(disp=-1)
    prediction3 = model_fit3.get_forecast(steps=test_exog.shape[0]).predicted_mean
    mean_absolute_percentage_error(test_set, prediction3)







