import datetime
from pathlib import Path

from matplotlib import pyplot as plt
from statsforecast.models import AutoARIMA, ARIMA
from statsforecast import StatsForecast
from fitting import get_stats, get_data_normalised
import pandas as pd
import numpy as np

if __name__=="__main__":


    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = get_data_normalised(data_folder)

    start = datetime.datetime(year=2015, month=1, day=1)
    end = start + datetime.timedelta(days=2*365)

    train_set = data[start:end]["total_demand"]
    train_exog = data[start: end].drop(["total_demand"], axis=1)
    #train_exog.drop(["sin_336_1", "cos_336_1", "holidays", "weekends", "rainfall", "solar_power"], inplace=True, axis=1)


    # use 1 day for the out of bag testing.
    test_set = data[end: end + datetime.timedelta(days=1)]["total_demand"]
    test_exog = data[end: end + datetime.timedelta(days=1)].drop("total_demand", axis=1)
    #test_exog.drop(["sin_336_1", "cos_336_1", "holidays", "weekends", "rainfall", "solar_power"], inplace=True, axis=1)

    #sf_fixed = StatsForecast(models=[ARIMA(order=(1, 0, 1), seasonal_order=(1, 1, 1) , season_length=48)],
    #                         freq='30min', n_jobs=-1)

    train_sf = pd.DataFrame({"unique_id": "total_demand", "ds": train_set.index, "y": train_set.values})
    train_sf = train_sf.merge(train_exog, left_on="ds", right_index=True, how="left")


    #sf_fixed.fit(df=train_sf)

    sf = StatsForecast(
        models=[AutoARIMA(
            season_length=48,
            max_p=2, max_q=2,
            max_P=2, max_Q=2,
            max_order=None,
            seasonal_test='ocsb',
        )],
        freq='30min',
        n_jobs=-1,
    )

    sf.fit(df=train_sf)


    #fitted = sf_fixed.fitted_[0][0].model_
    #print(fitted["bic"])

    #get_stats(sf_fixed, None)

    train_sf = pd.DataFrame({"unique_id": "total_demand", "ds": train_set.index, "y": train_set.values})
    #train_sf = train_sf.merge(train_exog[start:end], left_on="ds", right_index=True,how="left")
    #sf_fixed.fit(df=train_sf)

    fitted = sf.fitted_[0][0].model_
    print(fitted["aicc"])

    # Unpack the model information.
    fitted = sf.fitted_[0][0].model_
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


    test_exog["ds"] = test_set.index
    test_exog["unique_id"]= "total_demand"

    #forecasted_demand = sf_fixed.predict(49, X_df=test_exog)["ARIMA"].values
    #eval_data = data[end:end+datetime.timedelta(days=1)]

    #plt.plot(eval_data.index, eval_data["total_demand"])
    #plt.plot(eval_data.index, forecasted_demand)
    #plt.show()


    get_stats(sf, train_exog.columns)