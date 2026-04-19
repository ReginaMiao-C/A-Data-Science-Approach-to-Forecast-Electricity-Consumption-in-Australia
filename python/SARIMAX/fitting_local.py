"""
The purpose of this script is to determine the values of the SARIMAX parameters
"""
import itertools
import warnings
from multiprocessing import freeze_support, Pool

from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL

from statsforecast.models import AutoARIMA, ARIMA
from statsforecast import StatsForecast

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


def get_stats(model, exog_titles):
    """
    Try to recreate the statistical information that R produces

    args: model the fitted model with the information inside it
    exog_titles: titles to use for the exog varibles.
    """

    fitted = model.fitted_[0][0].model_
    coefs = np.array(list(fitted['coef'].values()))
    labels = list(fitted['coef'].keys())
    # get the variance-covariance matrix
    var_covar = fitted['var_coef']

    std_errors = np.sqrt(np.diag(var_covar))
    z_stats = coefs / std_errors

    # assume a two-tailed distribution
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

    print(f"\n{'':>10s} {'Coef':>10s}   {'Std Err':>5s}{'z':>10s}  {'p-value':>10s} |  {'Sig':>5s}")
    print("-" * 55)

    for enum, _ in enumerate(coefs):
        if labels[enum].startswith("ex"):
            key = int(labels[enum].split("_")[1]) - 1
            label = exog_titles[key]
        else:
            label = labels[enum]

        # use the R style significance display:
        if p_values[enum] < 0.001:
            sig = "***"
        elif p_values[enum] < 0.01:
            sig = "**"
        elif p_values[enum] < 0.05:
            sig = "*"
        elif p_values[enum] < 0.1:
            sig = "."
        else:
            sig = ""

        print(f"{label:>10s}:  {coefs[enum]:>10.4f}  {std_errors[enum]:>6.4f} {z_stats[enum]:>10.4f} {p_values[enum]:>10.4f} |  {sig:>5s}")


    print("\nSignificance codes:  *** 0.001  ** 0.01  * 0.05  . 0.1")


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


def get_data_normalised(data_folder):
    # load and use the datetime column to set the index:
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
    yearly_tersm = fourier_series(data.index, 365*48, K=1, t0=data.index[0])

    data["temp_1S"] = data["temperature"].shift(1).values*yearly_tersm["sin_17520_1"]
    data["temp_9S"] = data["temperature"].shift(9).values*yearly_tersm["sin_17520_1"]
    data["solar_4S"] = data["solar_power"].shift(4).values*yearly_tersm["sin_17520_1"]
    data["solar_16S"] = data["solar_power"].shift(16).values*yearly_tersm["sin_17520_1"]

    data["temp_1C"] = data["temperature"].shift(1).values*yearly_tersm["cos_17520_1"]
    data["temp_9C"] = data["temperature"].shift(9).values*yearly_tersm["cos_17520_1"]
    data["solar_4C"] = data["solar_power"].shift(4).values*yearly_tersm["cos_17520_1"]
    data["solar_16C"] = data["solar_power"].shift(16).values*yearly_tersm["cos_17520_1"]

    data["lag_48*7"] = data["total_demand"].shift(48 * 7)
    data = pd.concat([data, weekly_terms], axis=1)

    return data


def main():
    pass


def run_date_section(data, start, training_window, evaluation_window, using_exog=False):

    print(f"running:{start}")

    end = start + datetime.timedelta(days=training_window)
    eval_end = end + datetime.timedelta(days=evaluation_window)

    mask_train = (data.index >= start) & (data.index < end)
    mask_test = (data.index >= end) & (data.index < eval_end)

    training_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_train].index,
                                 "y": data[mask_train]["total_demand"].values})
    if using_exog:

        training_exog = data[mask_train].drop(["total_demand"], axis=1)
        training_set = training_set.merge(data[mask_train].drop(["total_demand"], axis=1), left_on="ds",
                                          right_index=True, how="left")


        testing_set = data[mask_test]
        testing_set = testing_set.drop(["total_demand"], axis=1)
        testing_set["unique_id"] = "total_demand"
        testing_set["ds"] = testing_set.index
    else:
        testing_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_test].index})

    model = ARIMA(order=(3, 1, 2), seasonal_order=(1, 0, 0), season_length=48)

    # Forecast for the out-of-bag testing, include transform back to real space (from log).
    if using_exog:
        testing_exog =  data[mask_test].drop(["total_demand"], axis=1)
        model.fit(y=training_set.y, X=training_exog.to_numpy())
        prediction = model.predict(h=48, level=[95], X=testing_exog.to_numpy())
    else:
        model.fit(y=training_set.y)
        prediction = model.predict(h=48, level=[95])

    # Unpack the model information.
    fitted = model.model_
    order = fitted['arma']  # (p, q, P, Q, s, d, D)
    arima_order = (order[0], order[5], order[1])
    seasonal_order = (order[2], order[6], order[3], order[4])
    aic = fitted['aic']
    loglik = fitted['loglik']
    coefs = fitted['coef']

    print(f"\n  Model  : ARIMA{arima_order}{seasonal_order}")
    print(f"  AIC    : {aic:.4f}")


    actuals = np.exp(data[mask_test]["total_demand"].values)
    forecast = np.exp(prediction["mean"])

    # calculate some values for the assessment of the model accuracy.
    mse = np.mean((actuals - forecast) ** 2)
    mape = mean_absolute_percentage_error(actuals, forecast)

    print(f"\n  MSE    : {mse:.4f}")
    print(f"  MAPE   : {mape:.4%}")

    # pack data for later analysis:
    results =  {"year": start.year, "month": start.month,
                    "train_start": start, "train_end": end, "train_obs": training_set.shape[0],
                    "arima_order": arima_order, "seasonal_order": seasonal_order,
                    "aic": aic, "loglik": loglik,
                    "coefs": coefs,
                    "mse_oob": mse, "mape_oob": mape}

    return results



def run_auto_fit(data):

    years = [2019]
    months = [1]

    # set the strides etc in days so we can use the index.
    training_window = 365
    evaluation_window = 1

    start_dates = [datetime.datetime(year=year, month=month, day=1) for (year, month) in itertools.product(years, months)]
    func_args = [(data, start, training_window, evaluation_window, True) for start in start_dates]

    # single threaded version:
    #for args in func_args:
    #    results = run_date_section(*args)

    with Pool(8) as pool:
        results = pool.starmap(run_date_section, func_args)

    df = pd.DataFrame(results)
    df.sort_values(by=["year", "month"], inplace=True)
    coef_df = df["coefs"].apply(pd.Series)
    coef_df.columns = [f"coef_{c}" for c in coef_df.columns]
    df = pd.concat([df.drop(columns="coefs"), coef_df], axis=1)
    """
    try:
        df.to_csv(root_folder/ "python"/ "SARIMAX" / f"analysis_results_with_exo_{datetime.date.today()}_generalmodel.csv", index=False)
    except Exception as e:
        print(e)
        df.to_csv(r"C:\Temp\file.csv", index=False)
        
    """



# Using the special variable
if __name__=="__main__":

    freeze_support()

    # not using the decomp
    decomposition = False
    sweep_no_exo = True

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = get_data_normalised(data_folder)


    # due to memory limitations will drop all the exogenous variables for the initial sweep;
    if sweep_no_exo:
        run_auto_fit(data)


    if decomposition:
        stl = STL(data["total_demand"].iloc[start:end], period=48)
        res = stl.fit()
        fig = res.plot()

        plt.show()


