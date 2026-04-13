"""
The purpose of this script is to determine the values of the SARIMAX parameters
"""
from itertools import product

from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL

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



def difference_testing(data, seasonal_value):
    adf_p = adfuller(data)[1]
    kpss_p = kpss(data, regression='c', nlags='auto')[1]

    print("ADF p:", adf_p, " | KPSS p:", kpss_p)

    data_diff = data.diff().dropna()

    adf_p = adfuller(data_diff)[1]
    kpss_p = kpss(data_diff, regression='c', nlags='auto')[1]

    print("ADF p:", adf_p, " | KPSS p:", kpss_p)

    data_seasonal = data.diff(seasonal_value).dropna()
    adf_p = adfuller(data_seasonal)[1]
    kpss_p = kpss(data_seasonal, nlags='auto')[1]

    print("ADF p:", adf_p, " | KPSS p:", kpss_p)


def main():
    pass

# Using the special variable
if __name__=="__main__":

    # not using the decomp
    decomposition = False

    # load and use the datetime column to set the index:
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

    years = [2018, 2019]
    months = [1, 3, 6, 9]

    results = []

    for year, month in product(years, months):

        print(year, month)

        # set stepping variables for the analysis:
        start = datetime.datetime(year=year, month=month, day=1)
        end = start + datetime.timedelta(days=7*8)

        test_set = data[end: end+datetime.timedelta(days=1)]["total_demand"]
        test_exog = data[end: end + datetime.timedelta(days=1)]["total_demand"]

        train_set = data[start:end]["total_demand"]
        train_exog = data[start:end].drop("total_demand", axis=1)

        autosolve = pm.auto_arima(y=train_set, exog=train_exog, m=48, trace=True, error_action="ignore", d=1, D=1,
                                  start_q=0, start_p=0, max_q=2, max_p=2,
                                  start_Q=0, start_P=0, max_Q=2, max_P=2, max_order=6, maxiter=200)

        forecast = autosolve.predict(n_periods=test_set.shape[0], exogenous=test_exog)

        results.append({"year": year, "month": month,
                        "mse_oob":mean_absolute_percentage_error(test_set, forecast),
                        "mape_oob":mean_absolute_percentage_error(test_set, forecast),
                        "aic": autosolve.aic(),
                        "order": autosolve.order,"seasonal_order": autosolve.seasonal_order})


    df = pd.DataFrame(results)
    df.to_csv(data_folder / "analysis_results.csv")

    if decomposition:
        stl = STL(data["total_demand"].iloc[start:end], period=48)
        res = stl.fit()
        fig = res.plot()

        plt.show()


