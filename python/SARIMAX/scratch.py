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



plot_acf(np.diff(seasonal.values), ax=axes[0], lags=97)
plot_pacf(np.diff(seasonal.values), ax=axes[1], lags=97)
plt.show()


esfe

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

testing_set = data[mask_test]
testing_set = testing_set.drop(["total_demand"], axis=1)
testing_set["unique_id"] = "total_demand"
testing_set["ds"] = testing_set.index

#testing_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_test].index})

models = StatsForecast(models=[ARIMA(order=(2, 1, 0), seasonal_order=(2, 1, 0), season_length=48)],
                       freq='30min', n_jobs=-1)

# run the fitting routine.
models.fit(df=training_set)
fitted = models.fitted_[0][0].model_


# Build a 1-step ahead exog DataFrame (just enough to satisfy h=1)
future_exog = pd.DataFrame({
    "unique_id": ["total_demand"],
    "ds": [training_set["ds"].iloc[-1] + pd.Timedelta("30min")],
    "your_exog_col": [0],   # dummy value, only needed to satisfy the call
})

exog_cols =['rainfall', 'holidays', 'weekends', 'pv_capacity', 'temperature', 'solar_power', 'temp_1S', 'temp_9S',
            'solar_4S', 'solar_16S', 'temp_1C', 'temp_9C', 'solar_4C', 'solar_16C', 'lag_48*7', 'sin_336_1', 'cos_336_1']

future_exog = pd.DataFrame({
    "unique_id": ["total_demand"],
    "ds": [training_set["ds"].iloc[-1] + pd.Timedelta("30min")],
    **{col: [0] for col in exog_cols}
})

models.forecast(df=training_set, X_df=future_exog, h=1, fitted=True)
fitted = models.forecast_fitted_values()

actuals = np.exp(data[mask_test]["total_demand"].values)
actual_fit = data[mask_train]["total_demand"].values


from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = fitted["y"] - fitted["ARIMA"]

lb_test = acorr_ljungbox(residuals, lags=[1, 24, 48, 96], return_df=True)
print(lb_test)


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

residuals = fitted["y"] - fitted["ARIMA"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram with normal overlay
axes[0].hist(residuals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
x = np.linspace(residuals.min(), residuals.max(), 200)
axes[0].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), "r--", linewidth=1.5, label="Normal")
axes[0].set_title("Residual distribution")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Density")
axes[0].legend()

# Q-Q plot to check normality more rigorously
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot")

plt.tight_layout()
plt.show()

# Summary stats
print(f"Mean:     {residuals.mean():.4f}  (should be ~0)")
print(f"Std:      {residuals.std():.4f}")
print(f"Skewness: {residuals.skew():.4f}")
print(f"Kurtosis: {residuals.kurtosis():.4f}  (excess, normal=0)")

test_exog = data[end: end + datetime.timedelta(days=1)].drop("total_demand", axis=1)
get_stats(models, test_exog.columns)