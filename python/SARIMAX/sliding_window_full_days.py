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
from fitting import get_data
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
    model = ARIMA(order=(2, 0, 0), seasonal_order=(0, 0, 0), season_length=48)

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


    results_data = {"model": model.model_["arma"],
                    "full_set_index": idx,
                    "full_set_mean":forecasted_demand,
                    "full_set_hi":forecasted_hi,
                    "full_set_lo":forecasted_lo,
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
                    "mape": mean_absolute_percentage_error(eval_data, forecasted_demand)}

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
    start = datetime.datetime(year=2016, month=1, day=1) - datetime.timedelta(days=training_window)
    start_dates = [start + datetime.timedelta(days=i) for i in list(range(0, 365*5, step))]
    func_args = [(data, start, training_window, evaluation_window, using_exog) for start in start_dates]

    #for args in func_args:
    #    results = run_date_section(*args)

    with Pool(8) as pool:
        results = pool.starmap(run_date_section, func_args)

    df = pd.DataFrame(results)
    df.sort_values(by="eval_date", inplace=True)
    df.to_csv(root_folder / "python" / "SARIMAX" / f"final_results_{datetime.date.today()}_with_exog_all_data.csv")

    exit()


    # old single threaded version
    for i in range(365):
        start += datetime.timedelta(days=step)

        print(f"Running: {start}")

        end = start + datetime.timedelta(days=training_window)
        eval_end = end + datetime.timedelta(days=evaluation_window)

        mask_train = (data.index >= start) & (data.index < end)
        mask_test = (data.index >= end) & (data.index < eval_end)

        training_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_train].index,
                                     "y": data[mask_train]["total_demand"].values})
        if using_exog:
            training_set = training_set.merge(data[mask_train].drop(["total_demand"], axis=1), left_on="ds",
                                          right_index=True, how="left")

        if using_exog:
            testing_set = data[mask_test]
            eval_data = testing_set["total_demand"].values
            testing_set = testing_set.drop(["total_demand"], axis=1)
            testing_set["unique_id"] = "total_demand"
            testing_set["ds"] = testing_set.index
        else:
            testing_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_test].index})

        models = StatsForecast(models=[ARIMA(order=(1, 0, 5), seasonal_order=(2, 1, 0), season_length=48)],
                                 freq='30min', n_jobs=-1)

        # run the fitting routine.
        models.fit(df=training_set)

        # extract the values for the assessment.
        forecasted_demand = models.predict(len(testing_set), testing_set)["ARIMA"].values
        eval_data =  data[mask_test]["total_demand"].values

        idx = testing_set.index.values
        if plot:
            insample_forecasts = models.fitted_[0, 0].predict_in_sample()["fitted"]
            plt.plot(training_set["ds"], insample_forecasts, color='grey')
            plt.plot(training_set["ds"],training_set["y"].values, color='k')

            plt.plot(idx, forecasted_demand, color='r')
            plt.plot(idx, eval_data, color='b')

        print(f"aic :{models.fitted_[0][0].model_["aic"]}")

        results_data = {"eval_date": end,
                        "aic":models.fitted_[0][0].model_["aic"],
                        "peak_actual": np.max(eval_data),
                        "peak_predicted": np.max(forecasted_demand),
                        "time_of_peak_actual": idx[np.argmax(eval_data)],
                        "time_of_peak_predicted": idx[np.argmax(forecasted_demand)],
                        "mse": mean_squared_error(eval_data, forecasted_demand),
                        "r2": r2_score(eval_data, forecasted_demand),
                        "mae": mean_absolute_error(eval_data, forecasted_demand),
                        "mape": mean_absolute_percentage_error(eval_data, forecasted_demand)}

        results.append(results_data)

    df = pd.DataFrame(results)
    df.to_csv(root_folder / "python" / "SARIMAX" / f"results_{datetime.date.today()}.csv")

    if plot:
        plt.show()













