import datetime
from itertools import product
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error
from statsforecast.models import AutoARIMA, ARIMA
from statsforecast import StatsForecast
from fitting import get_stats, get_data_normalised
import pandas as pd
import numpy as np
import matplotlib.dates as mdates


def plot_peaks(actual, hi, low, mean, x, name, axis_name):
    plt.style.use("seaborn-v0_8-whitegrid")  # nice default styling
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(x, low, hi, color="tab:blue", alpha=0.2,
                    label="Prediction interval")
    ax.plot(x, mean, color="tab:blue", linewidth=2, label="Predicted")

    ax.scatter(x, actual, color="black", s=40, alpha=0.8, label="Actual")

    # Labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{axis_name}")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()


    ax.legend()
    plt.tight_layout()

    plt.savefig(cwd / name, dpi=300)



cwd = Path.cwd()
no_exog = pd.read_csv(cwd / 'results_window_2026-04-19_with_exog.csv')

actual_peak_afternoon = np.expm1(no_exog['peak_actual_afternoon'])
pred_mean_peak_afternoon = np.expm1(no_exog['peak_predicted_afternoon_mean'])
pred_hi_peak_afternoon = np.expm1(no_exog['peak_predicted_afternoon_hi'])
pred_low_peak_afternoon = np.expm1(no_exog['peak_predicted_afternoon_lo'])

time = pd.to_datetime(no_exog['eval_date'], yearfirst=True)

#plot_peaks(actual_peak_afternoon, pred_hi_peak_afternoon, pred_low_peak_afternoon, pred_mean_peak_afternoon,
#           time, "peak_afternoon_with_exog", "Afternoon Peak Power (MW)")


actual_peak_morning = np.expm1(no_exog['peak_actual_morning'])
pred_mean_peak_morning = np.expm1(no_exog['peak_predicted_morning_mean'])
pred_hi_peak_morning = np.expm1(no_exog['peak_predicted_morning_hi'])
pred_low_peak_morning = np.expm1(no_exog['peak_predicted_morning_lo'])

time = pd.to_datetime(no_exog['eval_date'], yearfirst=True)

#plot_peaks(actual_peak_morning, pred_hi_peak_morning, pred_low_peak_morning, pred_mean_peak_morning,
#           time, "peak_morning_with_exog", "Morning Peak Power (MW)")


morning_residuals = actual_peak_afternoon - pred_mean_peak_morning

r2 = r2_score(actual_peak_morning, pred_mean_peak_morning)
rmse = root_mean_squared_error(actual_peak_morning, pred_mean_peak_morning)
mape = mean_absolute_percentage_error(actual_peak_morning, pred_mean_peak_morning)
hihi = np.count_nonzero(actual_peak_morning>pred_hi_peak_morning)
lowlow = np.count_nonzero(actual_peak_morning<pred_low_peak_morning)

plt.show()

print(r2,rmse, mape ,hihi, lowlow)

r2 = r2_score(actual_peak_afternoon, pred_mean_peak_afternoon)
rmse = root_mean_squared_error(actual_peak_afternoon, pred_mean_peak_afternoon)
mape = mean_absolute_percentage_error(actual_peak_afternoon, pred_mean_peak_afternoon)
hihi = np.count_nonzero(actual_peak_afternoon>pred_hi_peak_afternoon)
lowlow = np.count_nonzero(actual_peak_afternoon<pred_low_peak_afternoon)

afternoon_residuals = actual_peak_afternoon - pred_mean_peak_afternoon

plt.hist(afternoon_residuals, bins=20, label="Afternoon", color="orange", alpha=0.5)
plt.hist(morning_residuals, bins=20, label="Morning", color="red", alpha=0.5)

plt.show()

print(r2,rmse, mape ,hihi, lowlow)

