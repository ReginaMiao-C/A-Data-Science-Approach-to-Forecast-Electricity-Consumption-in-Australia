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
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.gofplots import qqplot

from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
from scipy.stats import norm

from coreforecast.scalers import boxcox_lambda, boxcox

from python.public_holidays import get_holidays

from python.colour_dict import *


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


def get_stats(model, exog_titles, _print=True):
    """
    Try to recreate the statistical information that R produces

    args: model the fitted model with the information inside it
    exog_titles: titles to use for the exog varibles.
    """

    try:
        fitted = model.fitted_[0][0].model_
    except:
        fitted = model.model_

    coefs = np.array(list(fitted['coef'].values()))
    labels = list(fitted['coef'].keys())
    # get the variance-covariance matrix
    var_covar = fitted['var_coef']

    std_errors = np.sqrt(np.diag(var_covar))
    z_stats = coefs / std_errors

    # assume a two-tailed distribution
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

    if _print:

        print(f"\n{'':>10s} {'Coef':>10s}   {'Std Err':>5s}{'z':>10s}  {'p-value':>10s} |  {'Sig':>5s}")
        print("-" * 55)

    results =  []

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

        if _print:

            print(f"{label:>10s}:  {coefs[enum]:>10.4f}  {std_errors[enum]:>6.4f} {z_stats[enum]:>10.4f} "
                  f"{p_values[enum]:>10.4f} |  {sig:>5s}")

        results.append({"label":label,"value":coefs[enum],"std_error":std_errors[enum],"z_score":z_stats[enum],
                        "p_value": p_values[enum],"significance":sig})

    if _print:
        print("\nSignificance codes:  *** 0.001  ** 0.01  * 0.05  . 0.1")

    return pd.DataFrame.from_dict(results)


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


def get_data(data_folder):
    # load and use the datetime column to set the index:
    data = pd.read_csv(data_folder / "all_data_30min.csv")
    data["datetime"] = pd.to_datetime(data["datetime"], yearfirst=True)
    data.index = data["datetime"]
    data.drop("datetime", axis=1, inplace=True)

    holidays = []
    for i in range(10):
        year = 2010 + i
        holidays.append(get_holidays(year))

    holidays = [dt for sublist in holidays for dt in sublist]

    one_hot_holidays = np.zeros_like(data.index, dtype=int)
    one_hot_weekdays = np.zeros_like(data.index, dtype=int)
    day_data = np.zeros_like(one_hot_holidays, dtype=int)
    month_data = np.zeros_like(one_hot_holidays, dtype=int)
    hour_data = np.zeros_like(one_hot_holidays, dtype=int)

    for enum, day_of_index in enumerate(data.index):

        temp_date = datetime.date(day_of_index.year, day_of_index.month, day_of_index.day)

        day_data[enum] = temp_date.weekday()
        month_data[enum] = temp_date.month
        hour_data[enum] = day_of_index.hour

        if temp_date in holidays:
            one_hot_holidays[enum] = 1

        if temp_date.weekday() == 5 or temp_date.weekday() == 6:
            one_hot_weekdays[enum] = 1

    weekly_terms = fourier_series(data.index, 7 * 48, K=1, t0=data.index[0])
    #yearly_tersm = fourier_series(data.index, 365*48, K=1, t0=data.index[0])

    data["temp_1"] = data["temperature"].shift(1).values#*yearly_tersm["sin_17520_1"]
    data["temp_9"] = data["temperature"].shift(9).values#*yearly_tersm["sin_17520_1"]
    data["solar_4"] = data["solar_power"].shift(4).values#*yearly_tersm["sin_17520_1"]
    data["solar_16"] = data["solar_power"].shift(16).values#*yearly_tersm["sin_17520_1"]
    data["Holidays"] = one_hot_holidays
    data["Weekends"] = one_hot_weekdays
    data["Day of the Week"] = day_data
    data["Month"] = month_data
    data["Hour"] = hour_data

    #data["temp_1C"] = data["temperature"].shift(1).values*yearly_tersm["cos_17520_1"]
    #data["temp_9C"] = data["temperature"].shift(9).values*yearly_tersm["cos_17520_1"]
    #data["solar_4C"] = data["solar_power"].shift(4).values*yearly_tersm["cos_17520_1"]
    #data["solar_16C"] = data["solar_power"].shift(16).values*yearly_tersm["cos_17520_1"]

    data["lag_48*7"] = data["total_demand"].shift(48 * 7)
    data = pd.concat([data, weekly_terms], axis=1)

    data.dropna(inplace=True, how="all", axis=0)

    return data


def main():
    pass


def run_date_section(data, start, training_window, evaluation_window, using_exog=False):

    print(f"running:{start}:{using_exog}")

    end = start + datetime.timedelta(days=training_window)
    eval_end = end + datetime.timedelta(days=evaluation_window)

    mask_train = (data.index >= start) & (data.index < end)
    mask_test = (data.index >= end) & (data.index < eval_end)

    training_set = data[mask_train]
    testing_set = data[mask_test]

    sf = AutoARIMA(
            season_length=48,
            seasonal_test="ocsb",
            test="kpss",
            max_p=7, max_q=7,
            max_P=7, max_Q=7,
            stepwise=True,max_order=20,
            trace=True, ic='aicc', nmodels=200)

    # catch a warning related to OCSB, it's annoying and the effect is not relevant for this report.
    with warnings.catch_warnings(action="ignore"):
        if using_exog:
            sf.fit(training_set["total_demand"], training_set.drop(columns=["total_demand"], axis=1).to_numpy())
        else:
            sf.fit(training_set["total_demand"])

    # Unpack the model information.
    fitted = sf.model_
    order = fitted['arma']  # (p, q, P, Q, s, d, D)
    arima_order = (order[0], order[5], order[1])
    seasonal_order = (order[2], order[6], order[3], order[4])
    aic = fitted['aic']
    loglik = fitted['loglik']
    coefs = fitted['coef']

    print(f"\n  Model  : ARIMA{arima_order}{seasonal_order}")
    print(f"  AIC    : {aic:.4f}")
    print(f"  Log-Lik: {loglik:.4f}")
    print(f"  Coefficients:")
    for k, v in coefs.items():
        print(f"    {k:>8s} = {v:.6f}")

    # Forecast for the out-of-bag testing, include transform back to real space (from log).
    if using_exog:
        forecast_sf = sf.predict(h=testing_set.shape[0],
                                 X=testing_set.drop(columns=["total_demand"], axis=1).to_numpy(),
                                 level=[95])
    else:
        forecast_sf = sf.predict(h=testing_set.shape[0],
                                 level=[95])

    forecast = forecast_sf["mean"]
    actuals =  data[mask_test]["total_demand"].values
    # calculate some values for the assessment of the model accuracy.
    mse = np.mean((actuals - forecast) ** 2)
    mape = mean_absolute_percentage_error(actuals, forecast)

    print(f"\n  MSE    : {mse:.4f}")
    print(f"  MAPE   : {mape:.4%}")

    # pack data for later analysis:
    results =  {"year": start.year, "month": start.month,
                #"box_cox_lambda":bcl,
                "exog": using_exog,
                    "train_start": start, "train_end": end, "train_obs": training_set.shape[0],
                    "arima_order": arima_order, "seasonal_order": seasonal_order,
                    "aic": aic, "loglik": loglik,
                    "coefs": coefs,
                    "mse_oob": mse, "mape_oob": mape}

    return results


def boxcox_backtransform_biasadj(fc_mean, fc_lower, fc_upper, lam):
    #Back-transform Box-Cox forecast with bias adjustment (https://robjhyndman.com/hyndsight/backtransforming/).

    # estimate the variance:
    z = norm.ppf(0.975)
    fvar = ((fc_upper - fc_lower) / (2 * z)) ** 2
    # calculate the mean
    mean_orig = np.power(lam * fc_mean + 1, 1 / lam)
    # adjust the mean.
    adjusted_mean = mean_orig * (1 + 0.5 * fvar * (1 - lam) / (mean_orig ** (2 * lam)))

    return adjusted_mean


def run_auto_fit(data):

    years = [2017,2018]
    months = [1,3,6,9]

    training_window = 8*7
    evaluation_window = 1

    func_args = [(data, datetime.datetime(year=year, month=month, day=1), training_window, evaluation_window, exog)
                 for (year, month, exog) in itertools.product(years, months, [True, False])]

    # single threaded version:
    #for args in func_args:
    #    results = run_date_section(*args)

    with Pool(16) as pool:
        results = pool.starmap(run_date_section, func_args)

    df = pd.DataFrame(results)
    df.sort_values(by=["year", "month"], inplace=True)
    coef_df = df["coefs"].apply(pd.Series)
    coef_df.columns = [f"coef_{c}" for c in coef_df.columns]
    df = pd.concat([df.drop(columns="coefs"), coef_df], axis=1)

    try:
        df.to_csv(root_folder/ "python"/ "SARIMAX" / f"analysis_results_with_no_scaling_{datetime.date.today()}_super_high.csv", index=False)
    except Exception as e:
        print(e)
        df.to_csv(r"C:\Temp\file.csv", index=False)



# Using the special variable
if __name__=="__main__":

    freeze_support()

    # not using the decomp
    decomposition = False
    sweep_no_exo = True
    bc_pics = False

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = get_data(data_folder)

    #reduce peaks in capacity KW->MW.
    data["pv_capacity"]= data["pv_capacity"]/1000
    #data.drop(columns=["lag_48*7", "pv_capacity"], inplace=True)

    if bc_pics:

        start = datetime.datetime(2018, 1, 1)
        end = start + datetime.timedelta(days=70)
        mask_train = (data.index >= start) & (data.index < end)

        bcl = boxcox_lambda(data["total_demand"][mask_train], method="loglik")

        data_to_plot = data["total_demand"][mask_train]

        data_to_plot_auto = boxcox(data_to_plot.values, bcl)
        data_to_plot_log = boxcox(data_to_plot.values, 0)

        fig, ax = plt.subplots(1, 2, figsize=(12,8))

        qqplot(data_to_plot, marker="o", color=demand_cols["all"], line="s", markerfacecolor=demand_cols["all"],
              markeredgecolor=demand_cols["all"], ax=ax[0])

        qqplot(data_to_plot_auto, marker="o", color=demand_cols["all"], line="s", markerfacecolor=demand_cols["all"],
               markeredgecolor=demand_cols["all"], ax=ax[1])

        ax[0].grid(True, axis='y', linestyle='--', alpha=0.5)
        ax[0].lines[1].set_color('black')
        ax[0].lines[1].set_linewidth(2)
        ax[0].title.set_text("Total Demand (Testing Data, Original)")

        ax[1].grid(True, axis='y', linestyle='--', alpha=0.5)
        ax[1].lines[1].set_color('black')
        ax[1].lines[1].set_linewidth(2)
        ax[1].set_title(f"Total Demand (Testing Data, Transformed Box-Cox ($\\lambda = {bcl:.3f}$))")

        plt.savefig(root_folder / "figures" /"qq_plot_masked_bc_transform_together.png")

        from scipy.stats import skew, kurtosis

        print(skew(data_to_plot), skew(data_to_plot_auto))
        print(kurtosis(data_to_plot), kurtosis(data_to_plot_auto))

    if sweep_no_exo:
        run_auto_fit(data)


    if decomposition:
        stl = STL(data["total_demand"].iloc[start:end], period=48)
        res = stl.fit()
        fig = res.plot()

        plt.show()


