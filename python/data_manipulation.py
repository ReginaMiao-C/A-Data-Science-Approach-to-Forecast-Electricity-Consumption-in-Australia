import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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


def get_data():
    # Use the relative paths for dat for this one.
    cwd = Path.cwd()
    root_folder = cwd.parent
    data_folder = root_folder / "data"

    # load data
    dispatched_power = pd.read_csv(data_folder / "totaldemand_nsw.csv")
    temperature_data = pd.read_csv(data_folder / "temperature_nsw.csv")
    solar_irradiance = pd.read_csv(data_folder / 'additional_data'  / "processed_data" / "daily_solar_exposure_bankstown.csv")
    rainfall = pd.read_csv(data_folder / 'additional_data'  / "processed_data" / "daily_rainfall_bankstown.csv")

    # set the dates into the datetime format:
    dispatched_power["DATE"] = pd.to_datetime(dispatched_power["DATETIME"], dayfirst=True).dt.date
    temperature_data["DATE"] = pd.to_datetime(temperature_data["DATETIME"], dayfirst=True).dt.date
    solar_irradiance["DATE"] = pd.to_datetime(solar_irradiance["DATE"], format="%Y-%m-%d").dt.date
    rainfall["DATE"] = pd.to_datetime(rainfall["DATE"], format="%Y-%m-%d").dt.date

    # get the min/max as required from the datasets.
    peak_power = dispatched_power.groupby("DATE").max()
    max_temperature = temperature_data.groupby("DATE")["TEMPERATURE"].max()
    min_temperature = temperature_data.groupby("DATE")["TEMPERATURE"].min()

    # Ensure datetime index
    peak_power.index = pd.to_datetime(peak_power.index)
    max_temperature.index = pd.to_datetime(max_temperature.index)
    min_temperature.index = pd.to_datetime(min_temperature.index)
    solar_irradiance.index = pd.to_datetime(solar_irradiance["DATE"])
    rainfall.index = pd.to_datetime(rainfall["DATE"])

    # find and fill any missing dates by using the average across the years:
    missing_dates = peak_power.index.difference(max_temperature.index)
    #print(missing_dates)
    max_temperature = fill_missing_values(max_temperature, missing_dates)

    missing_dates = peak_power.index.difference(min_temperature.index)
    #print(missing_dates)
    min_temperature = fill_missing_values(min_temperature, missing_dates)

    missing_dates = peak_power.index.difference(solar_irradiance.index)
    #print(missing_dates)
    solar_irradiance = fill_missing_values(solar_irradiance, missing_dates)

    missing_dates = peak_power.index.difference(rainfall.index)
    #print(missing_dates)
    rainfall = fill_missing_values(rainfall, missing_dates)

    data = pd.concat([rainfall["DAILY_RAINFALL"], solar_irradiance["DAILY_SOLAR_EXPOSURE"], min_temperature, max_temperature, peak_power["TOTALDEMAND"]], axis=1)
    data.columns = ["rainfall", "solar_exposure", "min_temperature", "max_temperature", "peak_power"]
    data.index = data.index.date
    return data


data = get_data()
data.hist(bins=20)
plt.show()

scaler = MinMaxScaler()
scaler.fit(data)

scaled_data = pd.DataFrame(scaler.transform(data))
scaled_data.hist(bins=20)
plt.show()

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data.corr())

lograin = pd.DataFrame.from_dict({"log(rain + 1)": np.log(data["rainfall"] + 1)})
lograin.hist(bins=20)
plt.show()