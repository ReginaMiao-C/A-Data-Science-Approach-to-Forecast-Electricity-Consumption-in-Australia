"""
Sliding window for the SARIMAX model using the previously determined parameter values
"""
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
import datetime
from statsforecast.models import ARIMA
import pandas as pd
from fitting import get_data, get_stats
import numpy as np
from multiprocessing import Pool, freeze_support


def run_date_section(data, start, training_window, evaluation_window, using_exog, parameters, full_day):

    """
    Runner for the ARIMA analysis
    :param data: Dataframe of data.
    :param start: start date of the training data as a datetime object
    :param training_window: training window size in days
    :param evaluation_window: evaluation window size in days (forecasted)
    :param using_exog: boolean for using exogenous parameters or not.
    :param parameters: dictionary of parameters for the ARIMA model.
    :param full_day: to save the full day of data.
    :return: dictionary of results.
    """
    print(f"running:{start}")

    end = start + datetime.timedelta(days=training_window)
    eval_end = end + datetime.timedelta(days=evaluation_window)

    mask_train = (data.index >= start) & (data.index < end)
    mask_test = (data.index >= end) & (data.index < eval_end)

    testing_set = data[mask_test]
    training_set = data[mask_train]
    # build the ARIMA model
    model = ARIMA(order=(parameters["p"], parameters["d"], parameters["q"]),
                  seasonal_order=(parameters["P"], parameters["D"], parameters["Q"]),
                  season_length=parameters["S"])

    if using_exog:
        training_exog = training_set.drop(["total_demand"], axis=1)
        testing_exog = testing_set.drop(["total_demand"], axis=1)
        model.fit(y=training_set["total_demand"].values, X=training_exog.to_numpy())
        prediction = model.predict(h=48, level=[95], X=testing_exog.to_numpy())
        stats = get_stats(model, exog_titles=testing_exog.columns, _print=False)

    else:
        model.fit(y=training_set["total_demand"].values)
        prediction = model.predict(h=48, level=[95])
        stats = get_stats(model, exog_titles=None, _print=False)


    # extract some info as it makes the later steps easier to understand.
    idx = testing_set.index.values
    eval_data = testing_set["total_demand"]
    forecasted_demand = prediction["mean"]
    forecasted_hi = prediction["hi-95"]
    forecasted_lo = prediction["lo-95"]

    # print some early info so we know how the model is going.
    print(f"{start}: {model.model_['aicc']}")

    # pack a dictionary with results.
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

    if full_day:
        results_data["full_set_index"] = idx
        results_data["full_set_mean"] = forecasted_demand
        results_data["full_set_hi"]= forecasted_hi
        results_data["full_set_lo"] = forecasted_lo

    return results_data

if __name__=="__main__":
    freeze_support()

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    save_folder = root_folder / "python" / "SARIMAX"
    save_folder.mkdir(parents=True, exist_ok=True)

    data = get_data(data_folder)
    data["pv_capacity"] = data["pv_capacity"]/1000

    using_exog = False
    # if the full days of data are wanted, i.e. every 30min data snap:
    full_days = False

    # leading name distinguishes daily max from all data.
    if full_days:
        leading_name = "interval"
    else:
        leading_name = "daily"

    if using_exog:
        parameters = {"p": 1, "q": 1, "P": 1, "Q": 1, "d": 0, "D": 0, "S": 48}
        file_name = f"{leading_name}_data_with_exog{datetime.date.today()}"
    else:
        parameters = {"p": 1, "q": 5, "P": 3, "Q": 0, "d": 0, "D": 1, "S": 48}
        file_name = f"{leading_name}_data_without_exog{datetime.date.today()}"

    # number of days to slide forward
    step = 1
    # set the strides etc in days so we can use the index.
    training_window = 7*8
    evaluation_window = 1

    cores = 20

    # start the date so the first prediction is the first day of the new year.
    start = datetime.datetime(year=2020, month=1, day=1) - datetime.timedelta(days=training_window)
    start_dates = [start + datetime.timedelta(days=i) for i in list(range(0, 365, step))]
    func_args = [(data, start, training_window, evaluation_window, using_exog, parameters,full_days) for start in start_dates]

    with Pool(cores) as pool:
        results = pool.starmap(run_date_section, func_args)

    # something odd happens with stats dictionary strip it out and save it separately.
    results_flat = [{k: v for k, v in r.items() if k != "stats"} for r in results]
    df = pd.DataFrame(results_flat)
    df.sort_values(by="eval_date", inplace=True)
    df.to_csv(save_folder / (file_name + ".csv"))

    stats_df = pd.concat([r["stats"].assign(eval_date=r["eval_date"]) for r in results])
    stats_df.sort_values(by="eval_date", inplace=True)
    stats_df.to_csv(save_folder / (file_name + "_stats.csv"))

    exit()













