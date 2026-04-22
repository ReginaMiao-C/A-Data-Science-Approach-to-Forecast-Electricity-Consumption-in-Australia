import datetime
from itertools import product
from pathlib import Path

from matplotlib import pyplot as plt
from statsforecast.models import AutoARIMA, ARIMA
from statsforecast import StatsForecast
from statsmodels.tsa.stattools import kpss

from fitting import get_stats, get_data_normalised, get_data
import pandas as pd
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda

if __name__=="__main__":


    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    #data = get_data_normalised(data_folder)
    data = get_data(data_folder)

    #l_bc = boxcox_lambda(data["total_demand"].values, season_length=48, method='loglik', lower=-10)
    #data["total_demand"] = boxcox(data["total_demand"], lmbda=0)

    import statsmodels.api as sm

    #sm.qqplot(data["total_demand"], line='s')  # 's' = standardised reference line
    #plt.show()

    print(data["total_demand"].value_counts().head(10))

    l_bc = boxcox_lambda(data["total_demand"].values, season_length=48, method='guerrero', lower=-10)
    data["total_demand"] = boxcox(data["total_demand"], lmbda=0)

    s=48
    log_y_sdiff = data["total_demand"][s:].values - data["total_demand"][:-s].values
    print(kpss(log_y_sdiff))

    plt.hist(data["total_demand"], bins=100)
    plt.show()

    sefsef

    #plt.plot(data["total_demand"])
    #plt.show()

    #data.drop(columns=['weekends', "holidays", "sin_336_1", "cos_336_1"], inplace=True)

    years = [2019]
    months = [12]

    p_values = {}

    for month in months:

        print(month)

        #start = datetime.datetime(year=year, month=month, day=1)
        #end = start + datetime.timedelta(days=365)
        #eval_end = end +   datetime.timedelta(days=1)

        eval_end = datetime.datetime(year=2019, month=2, day=27)
        end = eval_end - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=31*month)

        mask_train = (data.index >= start) & (data.index < end)
        mask_test = (data.index >= end) & (data.index < eval_end)

        train_set = data[start:end]["total_demand"]
        train_exog = data[start: end].drop(["total_demand"], axis=1)

        # use 1 day for the out of bag testing.
        test_set = data[mask_test]["total_demand"]
        test_exog = data[mask_test].drop("total_demand", axis=1)

        model = ARIMA(order=(0, 0, 0), seasonal_order=(0, 1, 0), season_length=48)

        # Forecast for the out-of-bag testing, include transform back to real space (from log).
        model.fit(y=train_set.values, X=train_exog.to_numpy())
        prediction = model.predict(h=48, level=[95], X=test_exog.to_numpy())

        fitted = model.model_
        #print(fitted["aicc"])

        # Unpack the model information.
        #order = fitted['arma']  # (p, q, P, Q, s, d, D)
        #arima_order = (order[0], order[5], order[1])
        #seasonal_order = (order[2], order[6], order[3], order[4])
        aic = fitted['aic']
        #loglik = fitted['loglik']
        #coefs = fitted['coef']

        #print(f"\n  Model  : ARIMA{arima_order}{seasonal_order}")
        print(f"  AIC    : {aic:.4f}")
        #print(f"  Log-Lik: {loglik:.4f}")
        #print(f"  Coefficients:")
        #for k, v in coefs.items():
        #    print(f"    {k:>8s} = {v:.6f}")


        #test_exog["ds"] = test_set.index
        #test_exog["unique_id"]= "total_demand"

        #forecasted_demand = sf_fixed.predict(49, X_df=test_exog)["ARIMA"].values
        #eval_data = data[end:end+datetime.timedelta(days=1)]

        #plt.plot(eval_data.index, eval_data["total_demand"])
        #plt.plot(eval_data.index, forecasted_demand)
        #plt.show()

        df = get_stats(model, train_exog.columns, _print=True)
        #p_values[f"{month}"] = df["p_value"]

    #results = pd.DataFrame.from_dict(p_values, orient='index')
    #results.columns = df["label"]

    #col_order = results.max().sort_values(ascending=False).index
    #results = results[col_order]

    #print(results)