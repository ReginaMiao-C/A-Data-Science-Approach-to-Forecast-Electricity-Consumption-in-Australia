"""
analysis using SARIMAX
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX
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

all_data = pd.read_csv(data_folder / "all_data_30min.csv")
all_data["date"] = pd.to_datetime(all_data["datetime"], yearfirst=True)

#subset = all_data.loc[all_data["date"] <= datetime.datetime(2010, 1, 14)]

trained_steps = 48*10

subset  = all_data.iloc[0:trained_steps, :]
forecast_steps = 48*2

print(subset.shape)

model_1 = SARIMAX(
    subset["total_demand"],
    exog=subset["temperature"],
    order=(2, 1, 1),
    seasonal_order=(0, 0, 0, 0)
)


fit_res1 = model_1.fit(disp=False, maxiter=500)
print(fit_res1.summary())

forecast = fit_res1.get_forecast(steps=forecast_steps, exog=all_data.iloc[trained_steps:(trained_steps+forecast_steps),:]["temperature"])
fitted = fit_res1.fittedvalues

#fig, ax = plt.subplots(figsize=(9,4))
#npre = 4
#ax.set(title='Peak Power', xlabel='Date', ylabel='Peak Power')
#subset.loc[:, 'total_demand'].plot(ax=ax, style='o', label='Observed')
#predict.predicted_mean.plot(ax=ax, style='r--', label='One-step-ahead forecast (+ temp)')

plt.plot(all_data.iloc[0:(trained_steps+forecast_steps),:]["date"], all_data.iloc[0:(trained_steps+forecast_steps),:]["total_demand"], color='blue')
plt.plot(subset["date"], fitted, color='red')
plt.plot(all_data.iloc[trained_steps:(trained_steps+forecast_steps),:]["date"] , forecast.predicted_mean)

plt.show()