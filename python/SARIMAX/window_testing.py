"""
The purpose of this script is to assess the effect of varying the window size.
"""
import itertools
from multiprocessing import freeze_support, Pool
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
from statsforecast.models import ARIMA
from fitting import get_data

def run_date_section(data, start, training_window, evaluation_window, using_exog=False):
    """
    Run the date section for the given model and date range, model parameters have been set by previous analysis.
    :param data: Dataframe of incoming data
    :param start: start date as Datetime
    :param training_window: size of the training window in days
    :param evaluation_window: size of the evaluation window in days
    :param using_exog: bool for using exogenous variables
    :return: list of dictionaries of fata
    """
    print(f"running:{start}")

    end = start + datetime.timedelta(days=training_window)
    eval_end = end + datetime.timedelta(days=evaluation_window)

    mask_train = (data.index >= start) & (data.index < end)
    mask_test = (data.index >= end) & (data.index < eval_end)

    training_set = data[mask_train]
    testing_set = data[mask_test]

    results = []

    try:
        if using_exog:
            model = ARIMA(order=(1, 0, 5), seasonal_order=(3, 1, 0), season_length=48)
            model.fit(training_set["total_demand"], training_set.drop(columns=["total_demand"], axis=1).to_numpy())
            forecast_sf = model.predict(h=testing_set.shape[0],
                                        X=testing_set.drop(columns=["total_demand"], axis=1).to_numpy(),
                                        level=[95])
        else:
            model = ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1), season_length=48)
            model.fit(training_set["total_demand"])
            forecast_sf = model.predict(h=testing_set.shape[0],level=[95])

        # Unpack the model information.
        fitted = model.model_
        aic = fitted['aicc']
        print(f"  AIC    : {aic:.4f}")

        actuals = data[mask_test]["total_demand"].values
        forecast = forecast_sf["mean"]
        # calculate some values for the assessment of the model accuracy.
        mse = np.mean((actuals - forecast) ** 2)

        try:
            mape = mean_absolute_percentage_error(actuals, forecast)
        except Exception as e:
            mape = np.nan

        print(f"\n  MSE    : {mse:.4f}")
        print(f"  MAPE   : {mape:.4%}")

        # pack data for later analysis:
        result =  {"year": start.year, "month": start.month,
                   "exog":using_exog,"eval_date": eval_end, "train_obs": training_set.shape[0],
                        "aic": aic, "mse_oob": mse, "mape_oob": mape}
    except Exception as e:
        print(e)
        result = {"year": start.year, "month": start.month,
                  "exog": using_exog, "eval_date": eval_end, "train_obs": training_set.shape[0],
                  "aic": np.nan, "mse_oob":  np.nan, "mape_oob":  np.nan}


    results.append(result)

    return results


def run(data, cores, save_location):
    """
    Runner for analysis of the window assessment
    :param data: incoming dataframe with demand and exogenous variables
    :param cores: number of cores to run none of <=1 is single core (good for debugging, otherwise slow).
    :param save_location: location to save the results
    :return: Nothing data is saved to the location.
    """

    years = [2017, 2018]
    months = [1, 3, 6, 9]
    windows = [1, 2, 4, 8, 16]
    exog = [True, False]

    evaluation_window = 1

    func_args = []

    for month, year, window, exog in itertools.product(months, years, windows, exog):

        eval_date = datetime.datetime(year, month, 1)
        end_date = eval_date - datetime.timedelta(days=evaluation_window)
        start_date = end_date - datetime.timedelta(days=window*7)

        func_args.append((data, start_date, 7*window, evaluation_window, exog))

    if cores is None or cores <= 1:
        for args in func_args:
            results = run_date_section(*args)
    else:
        with Pool(cores) as pool:
            results = pool.starmap(run_date_section, func_args)

    # flatten lists of lists (maybe just needed for single-threaded)
    results = [item for sublist in results for item in sublist]

    df = pd.DataFrame(results)
    df.sort_values(by=["year", "month"], inplace=True)
    try:
        df.to_csv(save_location /  f"window_assessment.csv", index=False)
    except Exception as e:
        # fail safe in case something is wrong with the original location
        print(e)
        df.to_csv(r"C:\Temp\file.csv", index=False)


if __name__=="__main__":
    freeze_support()

    cores = 20

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    save_location = root_folder/ "python"/ "SARIMAX" / "data"
    save_location.mkdir(parents=True, exist_ok=True)

    data = get_data(data_folder)
    # reduce by 1000 as it appears to help stability
    data["pv_capacity"] = data["pv_capacity"] / 1000
    run(data, cores, save_location)


