import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from fitting import get_data_normalised, get_stats

from statsmodels.tsa.stattools import kpss


def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="ct", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


from statsmodels.tsa.stattools import adfuller


def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

cwd = Path.cwd()
root_folder = cwd.parent.parent
data_folder = root_folder / "data"

data = get_data_normalised(data_folder)

energy = (data["total_demand"])
seasonal = energy.diff(periods=48).dropna()
#kpss_test(seasonal.values)
#adf_test(seasonal.values)

fig, axes = plt.subplots(1, 2)
fig.set_size_inches(10, 5)

#plot_acf(seasonal.values[1:4800], ax=axes[0], lags=97)
#plot_pacf(seasonal.values[1:4800], ax=axes[1], lags=97)
#plt.savefig(root_folder / "figures" / "pacf_seasonal.png")


#fig, axes = plt.subplots(1, 2)
#fig.set_size_inches(10, 5)

plot_acf(np.diff(seasonal.values[1:4800]), ax=axes[0], lags=97)
plot_pacf(np.diff(seasonal.values[1:4800]), ax=axes[1], lags=97)
plt.savefig(root_folder / "figures" / "pacf_seasonal_d_1.png")