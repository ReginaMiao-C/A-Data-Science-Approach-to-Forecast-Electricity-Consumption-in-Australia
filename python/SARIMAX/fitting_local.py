"""
The purpose of this script is to determine the values of the SARIMAX parameters
"""
import itertools
from multiprocessing import freeze_support, Pool

from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

from statsforecast.models import ARIMA

from fitting import get_data


def run_date_section(data, start, training_window, evaluation_window, using_exog=False):

    print(f"running:{start}")

    end = start + datetime.timedelta(days=training_window)
    eval_end = end + datetime.timedelta(days=evaluation_window)

    mask_train = (data.index >= start) & (data.index < end)
    mask_test = (data.index >= end) & (data.index < eval_end)

    training_set = data[mask_train]
    testing_set = data[mask_test]

    results = []

    for i,j,k,l in itertools.product([1], [5], [3], [0]):

        try:

            model = ARIMA(order=(i, 0, j), seasonal_order=(k, 1, l), season_length=48)

            if using_exog:
                model.fit(training_set["total_demand"], training_set.drop(columns=["total_demand"], axis=1).to_numpy())
                forecast_sf = model.predict(h=testing_set.shape[0],
                                            X=testing_set.drop(columns=["total_demand"], axis=1).to_numpy(),
                                            level=[95])
            else:
                model.fit(training_set["total_demand"])
                forecast_sf = model.predict(h=testing_set.shape[0],
                                            level=[95])

            # Unpack the model information.
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
            print(e)
            result = {"year": start.year, "month": start.month,
                      "train_start": start, "train_end": end, "train_obs": training_set.shape[0],
                      "arima_order": (i,0,j), "seasonal_order": (k,0,l),
                      "aic": np.nan, "loglik": np.nan,
                      "coefs": np.nan,
                      "mse_oob": np.nan, "mape_oob": np.nan}

        results.append(result)

    return results



def run_auto_fit(data):


    years = [2017, 2018]
    months = [1, 3, 6, 9]

    # set the strides etc in days so we can use the index.
    training_window = 7*8
    evaluation_window = 1

    start_dates = [datetime.datetime(year=year, month=month, day=1) for (year, month) in itertools.product(years, months)]
    func_args = [(data, start, training_window, evaluation_window, False) for start in start_dates]

    # single threaded version:
    #for args in func_args:
    #    results = run_date_section(*args)

    with Pool(8) as pool:
        results = pool.starmap(run_date_section, func_args)

    results = [item for sublist in results for item in sublist]

    df = pd.DataFrame(results)
    df.sort_values(by=["year", "month"], inplace=True)
    coef_df = df["coefs"].apply(pd.Series)
    coef_df.columns = [f"coef_{c}" for c in coef_df.columns]
    df = pd.concat([df.drop(columns="coefs"), coef_df], axis=1)
    try:
        df.to_csv(root_folder/ "python"/ "SARIMAX" / "data" / f"analysis_results_with_exo_{datetime.date.today()}_generalmodel.csv", index=False)
    except Exception as e:
        print(e)
        df.to_csv(r"C:\Temp\file.csv", index=False)



# Using the special variable
if __name__=="__main__":

    freeze_support()

    # not using the decomp
    decomposition = False
    sweep_no_exo = True

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = get_data(data_folder)
    # reduce peaks in capacity KW->MW.
    data["pv_capacity"] = data["pv_capacity"] / 1000

    # due to memory limitations will drop all the exogenous variables for the initial sweep;
    if sweep_no_exo:
        run_auto_fit(data)



