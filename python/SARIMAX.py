"""
analysis using SARIMAX
"""

from statsmodels.tsa.api import SARIMAX
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import MinMaxScaler


# Use the relative paths for dat for this one.
cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"

data = pd.read_csv(data_folder / "all_data.csv")
data["date"] = pd.to_datetime(data["date"], yearfirst=True).dt.date

data.drop(columns=["date"])

scaler = MinMaxScaler()
scaler.fit(data.iloc[:,1:])

scaled_data = scaler.transform(data.iloc[:,1:])

mod_ar1 = SARIMAX(data["peak_power"], exog=data[["max_temperature","min_temperature", "rainfall", "solar_exposure"]] ,order=(3,0,0))
fit_res1 = mod_ar1.fit(disp=False, maxiter=500)

mod_ar2 = SARIMAX(scaled_data[:,0], exog=scaled_data[:,1:] ,order=(3,0,0))
fit_res2 = mod_ar2.fit(disp=False, maxiter=500)
print(fit_res2.summary())


predict2 = fit_res2.get_prediction()
predict2.predicted_mean

# instad of repacking the data just pull the parts out of the scaler
predicted_mean2 = (predict2.predicted_mean - scaler.min_[4]) / scaler.scale_[4]

mod_ar3 = SARIMAX(data["peak_power"], order=(3,0,0))
fit_res3 = mod_ar2.fit(disp=False, maxiter=500)

print(fit_res1.summary())

predict = fit_res1.get_prediction()

fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Peak Power', xlabel='Date', ylabel='Peak Power')
data.loc[:, 'peak_power'].plot(ax=ax, style='o', label='Observed')
predict.predicted_mean.plot(ax=ax, style='r--', label='One-step-ahead forecast (+ temp)')
predict2.predicted_mean.plot(ax=ax, style='g--', label='One-step-ahead forecast')

plt.show()
