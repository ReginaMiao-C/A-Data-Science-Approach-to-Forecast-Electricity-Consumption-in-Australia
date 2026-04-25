"""
The purpose of this script is to determine the values of the SARIMAX parameters
"""
import itertools
import warnings
from multiprocessing import freeze_support, Pool

from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.gofplots import qqplot

from statsforecast.models import AutoARIMA, ARIMA
from statsforecast import StatsForecast

from coreforecast.scalers import boxcox_lambda, boxcox, inv_boxcox

from python.public_holidays import get_holidays

from fitting import get_data, get_stats
import numpy as np
from scipy.stats import norm

def boxcox_backtransform_biasadj(fc_mean, fc_lower, fc_upper, lam):
    #Back-transform Box-Cox forecast with bias adjustment (R-equivalent).

    # estimate the variance:
    z = norm.ppf(0.975)
    fvar = ((fc_upper - fc_lower) / (2 * z)) ** 2
    # calculate the mean
    mean_orig = np.power(lam * fc_mean + 1, 1 / lam)
    # adjust the mean.
    adjusted_mean = mean_orig * (1 + 0.5 * fvar * (1 - lam) / (mean_orig ** (2 * lam)))

    return adjusted_mean

cwd = Path.cwd()
root_folder = cwd.parent.parent
data_folder = root_folder / "data"
data = get_data(data_folder)


years = [2018]
months = [1]

# set the strides etc in days so we can use the index.
training_window =  8*8
evaluation_window = 1

data.drop(columns=["pv_capacity"], inplace=True)

start_dates = [datetime.datetime(year=year, month=month, day=1) for (year, month) in itertools.product(years, months)]
start = start_dates[0]

end = start + datetime.timedelta(days=training_window)
eval_end = end + datetime.timedelta(days=evaluation_window)

mask_train = (data.index >= start) & (data.index < end)
mask_test = (data.index >= end) & (data.index < eval_end)


bcl=boxcox_lambda(data["total_demand"], method = "guerrero", season_length=48)

plt.plot(boxcox(data["total_demand"], bcl))
plt.show()

sefesf

testing_set = data[mask_test]
exog = data[mask_train].drop(columns="total_demand")

training_set = pd.DataFrame({"unique_id": "total_demand", "ds": data[mask_train].index,
                                 "y": data[mask_train]["total_demand"].values})

#bcl = boxcox_lambda(training_set.y, method="loglik")
#training_set.y = boxcox(training_set.y,  bcl)

model = ARIMA(order=(1, 0, 0), seasonal_order=(0, 1, 0), season_length=48)
fit = model.fit(training_set.y, exog.to_numpy())

prediction = model.predict(h=48, level=[95], X=data[mask_test].drop(columns="total_demand").to_numpy())

fitted_vals = model.predict_in_sample()
#plt.plot(fitted_vals["fitted"])
#plt.plot(training_set.y)


get_stats(model, data[mask_test].drop(columns="total_demand").columns)


#plt.plot(data[mask_test].index, boxcox_backtransform_biasadj(prediction["mean"], prediction["hi-95"], prediction["lo-95"], bcl))
#plt.plot(data[mask_test].index, data[mask_test][["total_demand"]])
#plt.show()

