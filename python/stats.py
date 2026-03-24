"""
Scripting file to do some statistical analysis of the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Use the relative paths for dat for this one.
cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"

dispatched_power = pd.read_csv(data_folder / "totaldemand_nsw.csv")
dispatched_power["DATETIME"] = pd.to_datetime(dispatched_power["DATETIME"], dayfirst=True)

# split the datetime into some other units for ease of use:
dispatched_power["YEAR"] = dispatched_power["DATETIME"].dt.year
dispatched_power["MONTH"] = dispatched_power["DATETIME"].dt.month
dispatched_power["DATE"] = dispatched_power["DATETIME"].dt.date


temperature_data = pd.read_csv(data_folder / "temperature_nsw.csv")
temperature_data["DATETIME"] = pd.to_datetime(temperature_data["DATETIME"], dayfirst=True)
temperature_data["DATE"] = temperature_data["DATETIME"].dt.date


# don't trust the internal converse
t = (dispatched_power["DATETIME"] - dispatched_power["DATETIME"].min()).dt.total_seconds()

def integrate_group(g):
    t = (g["DATETIME"] - g["DATETIME"].min()).dt.total_seconds()
    return np.trapezoid(g["TOTALDEMAND"], x=t) / (3600*1000)  # GWhr


def total_energy():
    monthly_energy = dispatched_power[dispatched_power["YEAR"] < dispatched_power["YEAR"].max()].groupby(["YEAR","MONTH"]).apply(integrate_group)
    return monthly_energy

# https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
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


# https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


def fill_missing_values(df,missing_days, how="mean"):

    if how == "mean":
        # Create a Series to store filled values
        filled_temps = {}

        for day in missing_days:
            # Match same day & month across other years
            mask = (
                    (df.index.day == day.day) &
                    (df.index.month == day.month)
            )

            matched_values = df[mask]

            if len(matched_values) > 0:
                filled_temps[day] = matched_values.mean()
            else:
                matched_values[day] = np.nan  # fallback if no match

    # Add missing values into max_temp
    filled_series = pd.Series(filled_temps)
    df = pd.concat([df, filled_series]).sort_index()

    return df


adf_test(dispatched_power["TOTALDEMAND"])
kpss_test(dispatched_power["TOTALDEMAND"])
# both the tests suggest that the data is stationary well beyond the 1% Confidence Interval
peak_by_day = dispatched_power.groupby("DATE")["TOTALDEMAND"].max()
max_temp = temperature_data.groupby("DATE")["TEMPERATURE"].max()
min_temp = temperature_data.groupby("DATE")["TEMPERATURE"].min()

solar_irradiance = pd.read_csv(data_folder / 'additional_data'  / "processed_data" / "daily_solar_exposure_bankstown.csv")
solar_irradiance["DATE"] = pd.to_datetime(solar_irradiance["DATE"], dayfirst=True, format="%Y-%m-%d")
solar_irradiance.index = solar_irradiance["DATE"]

# Ensure datetime index
peak_by_day.index = pd.to_datetime(peak_by_day.index)
max_temp.index = pd.to_datetime(max_temp.index)
min_temp.index = pd.to_datetime(min_temp.index)

# Find missing days
missing_days = peak_by_day.index.difference(max_temp.index)

# Create a Series to store filled values
max_temp = fill_missing_values(max_temp, missing_days)
min_temp = fill_missing_values(min_temp, missing_days)

data = pd.DataFrame.from_dict({"Max_Temperature": max_temp, "Max_Demand":peak_by_day, "Min_Temperature":min_temp,
                               "Solar_Irradiance": solar_irradiance["DAILY_SOLAR_EXPOSURE"]})
data.corr(method="pearson")
data.corr(method="spearman")
"""
plt.figure(figsize=(8,6))
plt.scatter(data["Max_Temperature"], data["Max_Demand"], alpha=0.5)
plt.xlabel("Max Temperature")
plt.ylabel("Peak Demand")
plt.title("Peak Demand vs Max Temperature")
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(data["Min_Temperature"], data["Max_Demand"], alpha=0.5)
plt.xlabel("Min Temperature")
plt.ylabel("Peak Demand")
plt.title("Peak Demand vs Min Temperature")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(data["Solar_Irradiance"], data["Max_Demand"], alpha=0.5)
plt.xlabel("Solar Irradiance")
plt.ylabel("Peak Demand")
plt.title("Peak Demand vs Solar Irradiance")
plt.show()
"""


data["temp_centered"] = data["Max_Temperature"] - data["Max_Temperature"].mean()
data["temp_centered_sq"] = data["temp_centered"]**2

X = sm.add_constant(data[["temp_centered", "temp_centered_sq"]])
y = data["Max_Demand"]

model = sm.OLS(y, X).fit()
print(model.summary())

plot_acf(data["Max_Demand"], lags=31)
plt.title("Autocorrelation of Peak Demand")
plt.show()