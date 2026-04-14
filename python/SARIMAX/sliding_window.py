"""
Sliding window for the SARIMAX model using the previously determined parameter values
"""
from pathlib import Path
import pandas as pd
from fontTools.varLib.instancer.names import ELIDABLE_AXIS_VALUE_NAME
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
import datetime
from statsforecast.models import AutoARIMA, ARIMA
from statsforecast import StatsForecast
import pandas as pd
from fitting import get_data_normalised, get_stats
import numpy as np


if __name__=="__main__":

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = get_data_normalised(data_folder)

    plot = False
    using_exog = True

    # number of days to slide forward
    step = 1

    # set the strides etc in days so we can use the index.
    start = datetime.datetime(year=2020, month=1, day=1)
    training_window = 7*8
    evaluation_window = 1
    results = []

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

        models = StatsForecast(models=[ARIMA(order=(2, 1, 0), seasonal_order=(2, 1, 0), season_length=48)],
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

        results_data = {"eval_date": end,
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













