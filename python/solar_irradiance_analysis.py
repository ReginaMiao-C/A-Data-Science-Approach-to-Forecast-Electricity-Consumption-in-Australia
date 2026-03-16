import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy

# Use the relative paths for dat for this one.
cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"

if not data_folder.exists():
    raise FileNotFoundError("Data folder does not exist, check you've updated everything")

solar_data = pd.read_csv(data_folder / "POWER_Point_Hourly_20100101_20260316_033d91S_151d06E_UTC.csv", skiprows=10)
temperature_data = pd.read_csv(data_folder / "temperature_nsw.csv")
temperature_data["DATETIME"] = pd.to_datetime(temperature_data["DATETIME"], dayfirst=True)

# check for missing data (-999 according to the file information)

missingdata_count = solar_data["ALLSKY_SFC_SW_DWN"].value_counts().get(-999, 0)

# 1800 missing points, lets' see where
print(f"Earliest year data is missing  {solar_data[solar_data["ALLSKY_SFC_SW_DWN"] == -999]["YEAR"].min()}")
# nevermind, it's all from this year.

# convert the time/data columns into a single DATETIME (Going to assume time in temp file is in local (AEST, no daylight
# savings)
UTC_to_AEST = 10

solar_data["DATETIME"] = pd.to_datetime({"year": solar_data["YEAR"], "month": solar_data["MO"],
                                         "day": solar_data["DY"],"hour": solar_data["HR"]})

# now add on the UTC correction so the time properly rolls over.
solar_data["DATETIME"] += datetime.timedelta(hours=UTC_to_AEST)

NASA_Temps = np.interp(x=temperature_data["DATETIME"], xp=solar_data["DATETIME"], fp=solar_data["T2M"])

# plot a snap shot
plt.plot(temperature_data["DATETIME"][100:1000], NASA_Temps[100:1000])
plt.plot(temperature_data["DATETIME"][100:1000], temperature_data["TEMPERATURE"][100:1000])
plt.show()
# we note that the values all line well.

# Check using Pearson R (assuming linear relationship and normally distributed data for now).
pearson_r2 = scipy.stats.pearsonr(temperature_data["TEMPERATURE"], NASA_Temps)

# now happy with the time sequencing:
NASA_SIR = np.interp(x=temperature_data["DATETIME"], xp=solar_data["DATETIME"], fp=solar_data["ALLSKY_SFC_SW_DWN"])

pearson_r2 = scipy.stats.kendalltau(solar_data["ALLSKY_SFC_SW_DWN"], solar_data["T2M"])


plt.plot(temperature_data["DATETIME"][100:1000], NASA_Temps[100:1000])
plt.plot(temperature_data["DATETIME"][100:1000], NASA_SIR[100:1000])
plt.show()

