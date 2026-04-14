import datetime
from pathlib import Path

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

    start = datetime.datetime(year=2019, month=9, day=1)
    end = start + datetime.timedelta(days=7 * 8)

    train_set = data[start:end]["total_demand"]
    train_exog = data[start: end].drop(["total_demand"], axis=1)
    # use 1 day for the out of bag testing.
    test_set = data[end: end + datetime.timedelta(days=1)]["total_demand"]
    test_exog = data[end: end + datetime.timedelta(days=1)].drop("total_demand", axis=1)

    sf_fixed = StatsForecast(models=[ARIMA(order=(2, 1, 0), seasonal_order=(2, 1, 0) , season_length=48)],
                             freq='30min', n_jobs=-1)

    train_sf = pd.DataFrame({"unique_id": "total_demand", "ds": train_set.index, "y": train_set.values})
    #sf_fixed.fit(df=train_sf)

    #fitted = sf_fixed.fitted_[0][0].model_
    #print(fitted["bic"])

    #get_stats(sf_fixed, None)

    train_sf = pd.DataFrame({"unique_id": "total_demand", "ds": train_set.index, "y": train_set.values})
    train_sf = train_sf.merge(train_exog[start:end], left_on="ds", right_index=True,how="left")
    sf_fixed.fit(df=train_sf)

    fitted = sf_fixed.fitted_[0][0].model_
    print(fitted["bic"])

    get_stats(sf_fixed, train_exog.columns)