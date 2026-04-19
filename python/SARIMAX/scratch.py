import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

import seaborn as sns

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
start = datetime.datetime(2018, 8, 1)
training_window = 365
evaluation_window = 1

print(f"running:{start}")

end = start + datetime.timedelta(days=training_window)
eval_end = end + datetime.timedelta(days=evaluation_window)

mask_train = (data.index >= start) & (data.index < end)
mask_test = (data.index >= end) & (data.index < eval_end)

training_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_train].index,
                                 "y": data[mask_train]["total_demand"].values})

training_set = training_set.merge(data[mask_train].drop(["total_demand"], axis=1), left_on="ds",
                                          right_index=True, how="left")


training_exog = data[mask_train].drop(["total_demand"], axis=1)

testing_set = data[mask_test]
test_set_values = testing_set["total_demand"]
testing_set = testing_set.drop(["total_demand"], axis=1)
#testing_set["unique_id"] = "total_demand"
#testing_set["ds"] = testing_set.index

#testing_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_test].index})

model = ARIMA(order=(2, 1, 0), seasonal_order=(2, 1, 0), season_length=48)
# run the fitting routine.
model.fit(y=training_set.y, X=training_exog.to_numpy())
prediction = model.predict(h=48, level=[95], X=testing_set.values)
in_sample_values = model.predict_in_sample()

actuals = np.exp(data[mask_test]["total_demand"].values)
actual_fit = data[mask_train]["total_demand"].values

from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = training_set.y - in_sample_values["fitted"]

prediction["time"] = data[mask_test].index

fig, ax1 = plt.subplots()
# First axis (Demand)
sns.lineplot(data=test_set_values, ax=ax1)
sns.lineplot(data=prediction, x="time", y="mean" , ax=ax1, color="blue")
sns.lineplot(data=prediction, x="time", y="lo-95", ax=ax1, color="blue", alpha=0.3)
sns.lineplot(data=prediction, x="time", y="hi-95", ax=ax1, color="blue", alpha=0.3)
lines = ax1.get_lines()
plt.fill_between(lines[1].get_xdata(), lines[2].get_ydata(), lines[3].get_ydata(), color='blue', alpha=0.1)
ax1.set_ylabel("Power")



