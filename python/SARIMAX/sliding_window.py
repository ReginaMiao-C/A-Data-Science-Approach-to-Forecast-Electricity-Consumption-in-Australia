"""
Sliding window for the SARIMAX model using the previously determined parameter values
"""
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
import datetime
from statsforecast.models import ARIMA
from statsforecast import StatsForecast
import pandas as pd
from fitting import get_data, get_stats
import numpy as np
from multiprocessing import Pool, freeze_support


def run_date_section(data, start, training_window, evaluation_window, using_exog=True):

    print(f"running:{start}")

    end = start + datetime.timedelta(days=training_window)
    eval_end = end + datetime.timedelta(days=evaluation_window)

    mask_train = (data.index >= start) & (data.index < end)
    mask_test = (data.index >= end) & (data.index < eval_end)

    testing_set = data[mask_test]
    training_set = data[mask_train]
    # build the ARIMA model
    model = ARIMA(order=(1, 0, 5), seasonal_order=(3, 1, 0), season_length=48)

    if using_exog:
        training_exog = training_set.drop(["total_demand"], axis=1)
        testing_exog = testing_set.drop(["total_demand"], axis=1)
        model.fit(y=training_set["total_demand"].values, X=training_exog.to_numpy())
        prediction = model.predict(h=48, level=[95], X=testing_exog.to_numpy())
    else:
        model.fit(y=training_set["total_demand"].values)
        prediction = model.predict(h=48, level=[95])

    idx = testing_set.index.values
    eval_data = testing_set["total_demand"]
    forecasted_demand = prediction["mean"]
    forecasted_hi = prediction["hi-95"]
    forecasted_lo = prediction["lo-95"]

    print(f"{start}: {model.model_['aicc']}")

    if using_exog:
        stats = get_stats(model, exog_titles=testing_exog.columns, _print=False)
    else:
        stats = get_stats(model, exog_titles=None, _print=False)


    results_data = {"model": model.model_["arma"],
                    "traing_window size": training_set.shape[0],
                    "using_exog": using_exog,
                    "eval_date": end,
                    "model aicc": model.model_['aicc'],
                    "peak_actual_afternoon": np.max(eval_data[25:48]),
                    "peak_predicted_afternoon_mean": np.max(forecasted_demand[25:48]),
                    "peak_predicted_afternoon_hi": np.max(forecasted_hi[25:48]),
                    "peak_predicted_afternoon_lo": np.max(forecasted_lo[25:48]),
                    "time_of_peak_actual_afternoon": idx[np.argmax(eval_data[25:48])],
                    "time_of_peak_predicted_afternoon": idx[np.argmax(forecasted_demand[25:48])],
                    "peak_actual_morning": np.max(eval_data[0:25]),
                    "peak_predicted_morning_mean": np.max(forecasted_demand[0:25]),
                    "peak_predicted_morning_hi": np.max(forecasted_hi[0:25]),
                    "peak_predicted_morning_lo": np.max(forecasted_lo[0:25]),
                    "time_of_peak_actual_morning": idx[np.argmax(eval_data[0:25])],
                    "time_of_peak_predicted_morning": idx[np.argmax(forecasted_demand[0:25])],
                    "mse": mean_squared_error(eval_data, forecasted_demand),
                    "r2": r2_score(eval_data, forecasted_demand),
                    "mae": mean_absolute_error(eval_data, forecasted_demand),
                    "mape": mean_absolute_percentage_error(eval_data, forecasted_demand),
                    "stats":  stats}

    return results_data

if __name__=="__main__":
    freeze_support()

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = get_data(data_folder)
    data["pv_capacity"] = data["pv_capacity"]/1000

    plot = False
    using_exog = True

    # number of days to slide forward
    step = 1
    # set the strides etc in days so we can use the index.
    training_window = 7*8
    evaluation_window = 1

    # start the date so the first prediction is the first day of the new year.
    start = datetime.datetime(year=2020, month=1, day=1) - datetime.timedelta(days=training_window)
    start_dates = [start + datetime.timedelta(days=i) for i in list(range(0, 365, step))]
    func_args = [(data, start, training_window, evaluation_window, using_exog) for start in start_dates]

    #for args in func_args:
    #    results = run_date_section(*args)

    with Pool(20) as pool:
        results = pool.starmap(run_date_section, func_args)

    export_file_name =  f"daily_data_without_exog{datetime.date.today()}"

    # something odd happens with stats dictionary strip it out and save it separately.
    results_flat = [{k: v for k, v in r.items() if k != "stats"} for r in results]
    df = pd.DataFrame(results_flat)
    df.sort_values(by="eval_date", inplace=True)
    df.to_csv(root_folder / "python" / "SARIMAX" / (export_file_name + ".csv"))

    stats_df = pd.concat([r["stats"].assign(eval_date=r["eval_date"]) for r in results])
    stats_df.sort_values(by="eval_date", inplace=True)
    stats_df.to_csv(root_folder / "python" / "SARIMAX" / (export_file_name + "_stats.csv"))

    exit()













