"""
Testing of the Arima process, doesn't use the AUTOARIMA so need to supply you're own parameters,
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


def run_date_section(data, start, training_window, evaluation_window, using_exog, parameters):
    """
    Runs the ARIMA assessment for the given model
    :param data: Dataframe of incoming data
    :param start: start date as a Datetime object
    :param training_window: number of days between start and end date of the training window
    :param evaluation_window: numbers of days for the evaluation window (forecasted)
    :param using_exog: bool for using exogenous parameters
    :param parameters: dictionary of parameters for the ARIMA model.
    :return: list of dictionaries of results data.
    """
    print(f"running:{start}")

    end = start + datetime.timedelta(days=training_window)
    eval_end = end + datetime.timedelta(days=evaluation_window)

    mask_train = (data.index >= start) & (data.index < end)
    mask_test = (data.index >= end) & (data.index < eval_end)

    training_set = data[mask_train]
    testing_set = data[mask_test]

    # unpack the varibles because it's a little neater/easier to read.
    list_p = parameters["p"]
    list_q = parameters["q"]
    list_P = parameters["P"]
    list_Q = parameters["Q"]
    d = parameters["d"]
    D = parameters["D"]
    S = parameters["S"]

    results = []

    for p,q,P,Q in itertools.product(list_p, list_q, list_P, list_Q):

        try:

            model = ARIMA(order=(p, d, q), seasonal_order=(P, D, Q), season_length=S)

            if using_exog:
                model.fit(training_set["total_demand"], training_set.drop(columns=["total_demand"], axis=1).to_numpy())
                forecast_sf = model.predict(h=testing_set.shape[0],
                                            X=testing_set.drop(columns=["total_demand"], axis=1).to_numpy(),
                                            level=[95])
            else:
                model.fit(training_set["total_demand"])
                forecast_sf = model.predict(h=testing_set.shape[0],
                                            level=[95])

            # Unpack the model information
            fitted = model.model_
            order = fitted['arma']  # (p, q, P, Q, s, d, D)
            arima_order = (order[0], order[5], order[1])
            seasonal_order = (order[2], order[6], order[3], order[4])
            aic = fitted['aicc']
            loglik = fitted['loglik']
            coefs = fitted['coef']

            print(f"\n  Model  : ARIMA{arima_order}{seasonal_order}")
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
                            "train_start": start, "train_end": end, "train_obs": training_set.shape[0],
                            "arima_order": arima_order, "seasonal_order": seasonal_order,
                            "aic": aic, "loglik": loglik,
                            "coefs": coefs,
                            "mse_oob": mse, "mape_oob": mape}
        except Exception as e:
            # collect data incase one of the cases fails.
            print(e)
            result = {"year": start.year, "month": start.month,
                      "train_start": start, "train_end": end, "train_obs": training_set.shape[0],
                      "arima_order": (p,0,q), "seasonal_order": (P,0,Q),
                      "aic": np.nan, "loglik": np.nan,
                      "coefs": np.nan,
                      "mse_oob": np.nan, "mape_oob": np.nan}

        results.append(result)

    return results



def run_auto_fit(data, cores, save_location,filename, parameters, using_exog):
    """
    Runner for the ARIMA process.
    :param data: Dataframe of data
    :param cores: number of cores to use
    :param save_location: location to save the data into
    :param filename: filename of the data
    :param parameters: dictionary of parameters.
    :param using_exog: boolean for using exogenous variables,
    :return: None, data is saved
    """

    # hard coded data for the assessment
    years = [2017, 2018]
    months = [1, 3, 6, 9]

    # set the strides etc in days so we can use the index.
    training_window = 7*8
    evaluation_window = 1

    start_dates = [datetime.datetime(year=year, month=month, day=1) for (year, month) in itertools.product(years, months)]
    func_args = [(data, start, training_window, evaluation_window, using_exog, parameters) for start in start_dates]

    # single threaded version:

    if cores is None or cores < 2:
        results = []

        for args in func_args:
            result = run_date_section(*args)
            results.append(result)
    else:
        with Pool(cores) as pool:
            results = pool.starmap(run_date_section, func_args)

    # flatten the array if needed
    results = [item for sublist in results for item in sublist]

    df = pd.DataFrame(results)
    df.sort_values(by=["year", "month"], inplace=True)
    coef_df = df["coefs"].apply(pd.Series)
    coef_df.columns = [f"coef_{c}" for c in coef_df.columns]
    df = pd.concat([df.drop(columns="coefs"), coef_df], axis=1)
    try:
        df.to_csv(save_location / f"{filename}.csv", index=False)
    except Exception as e:
        print(e)
        df.to_csv(r"C:\Temp\file.csv", index=False)



# Using the special variable
if __name__=="__main__":

    freeze_support()

    cores = 20

    run = "no_exog_gmod"

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    save_location = root_folder/ "python"/"SARIMAX" / "data"
    save_location = Path(r"C:/Temp")
    data = get_data(data_folder)
    # reduce peaks in capacity, seems to make it run faster.
    data["pv_capacity"] = data["pv_capacity"] / 1000

    # parameter setting for the different models ran as part of the assessment
    if run == "parsweep":
        parameters = {"p":[0,1,2], "q":[0,1,2], "P":[0,1,2], "Q":[0,1,2], "d":0, "D":0, "S":48}
        file_name = f"analysis_results_with_exo_{datetime.date.today()}_generalmodel_parsweep"
        exog = True
    elif run == "exog_gmod":
        parameters = {"p": [1], "q": [5], "P": [3], "Q": [0], "d": 0, "D": 1, "S": 48}
        file_name = f"analysis_results_with_exo_{datetime.date.today()}_generalmodel"
        exog = True
    elif run == "no_exog_gmod":
        parameters = {"p": [1], "q": [5], "P": [3], "Q": [0], "d": 0, "D": 1, "S": 48}
        file_name = f"analysis_results_without_exo_{datetime.date.today()}_generalmodel"
        exog = False
    else:
        # just exit if a parameter hasn't been provided. (you can provide your own if you want).
        exit("Invalid run")

    run_auto_fit(data=data, cores=cores, save_location=save_location, filename=file_name,
                 parameters=parameters, using_exog=exog)



