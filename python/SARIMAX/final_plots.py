from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import python.colour_dict as colour_dict
from python.public_holidays import get_holidays
from fitting import get_data

def plot_peaks(actual, hi, low, mean, x, name, axis_name, plt_title=None):
    plt.style.use("seaborn-v0_8-whitegrid")  # nice default styling
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(x, low, hi, color="tab:blue", alpha=0.2,
                    label="Prediction interval")
    ax.plot(x, mean, color="tab:blue", linewidth=2, label="Predicted")
    ax.scatter(x, actual, color=colour_dict.demand_cols['peak'], s=40, alpha=0.8, label="Actual")

    # Labels and title
    ax.set_xlabel("Time")
    ax.set_title(plt_title)
    ax.set_ylabel(f"{axis_name}")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    ax.legend()
    plt.tight_layout()
    plt_folder = cwd.parent.parent / "figures"
    plt.savefig(plt_folder / name, dpi=300)

cwd = Path.cwd()

plt_folder = cwd.parent / "figures"

#no_exog = pd.read_csv(cwd / 'final_results_window_2026-04-22_with_exog.csv')
no_exog = pd.read_csv(cwd / 'final_results_window_2026-04-23_with_exog_low_aic.csv')

cwd = Path.cwd()
root_folder = cwd.parent.parent
data_folder = root_folder / "data"
data = get_data(data_folder)
data["date"] = pd.to_datetime(data.index)
data["date"] = data["date"].dt.date

max_daily_temp = data.groupby("date").max()["temperature"]

# one bad value is noted, just remove it
no_exog.dropna(how='any',axis=0, inplace=True)
time = pd.to_datetime(no_exog['eval_date'], yearfirst=True)

filtered_temp = max_daily_temp[max_daily_temp.index.isin(time.dt.date)]


actual_peak_afternoon = (no_exog['peak_actual_afternoon'])
pred_mean_peak_afternoon = (no_exog['peak_predicted_afternoon_mean'])
pred_hi_peak_afternoon = (no_exog['peak_predicted_afternoon_hi'])
pred_low_peak_afternoon = (no_exog['peak_predicted_afternoon_lo'])

actual_peak_morning = (no_exog['peak_actual_morning'])
pred_mean_peak_morning = (no_exog['peak_predicted_morning_mean'])
pred_hi_peak_morning = (no_exog['peak_predicted_morning_hi'])
pred_low_peak_morning = (no_exog['peak_predicted_morning_lo'])

#plot_peaks(actual_peak_morning, pred_hi_peak_morning, pred_low_peak_morning, pred_mean_peak_morning,
#           time, "peak_morning_with_exog", "Morning Peak Power (MW)")

actual_peak = np.max([actual_peak_afternoon, actual_peak_morning],axis=0)

daily_peaks_arg = np.argmax([pred_mean_peak_afternoon, pred_mean_peak_morning], axis=0)

daily_peaks = np.array([pred_mean_peak_afternoon, pred_mean_peak_morning])
daily_peaks= daily_peaks[daily_peaks_arg, np.arange(daily_peaks.shape[1])]


daily_hi = np.array([pred_hi_peak_afternoon, pred_hi_peak_morning])
daily_hi= daily_hi[daily_peaks_arg, np.arange(daily_hi.shape[1])]

daily_lo = np.array([pred_low_peak_morning, pred_low_peak_afternoon])
daily_lo= daily_lo[daily_peaks_arg, np.arange(daily_lo.shape[1])]


plot_peaks(actual_peak,daily_hi, daily_lo, daily_peaks,
           time, "SARIMAX_peak_daily", "Peak Electricity Demand (MW)", plt_title="SARIMAX")

holidays = get_holidays(2020)
holidays = pd.to_datetime(holidays)

residuals = actual_peak - daily_peaks

print(f"RMSE: {root_mean_squared_error(actual_peak, daily_peaks)}")
print(f"MAPE: {mean_absolute_percentage_error(actual_peak, daily_peaks)}")
print(f"R2: {r2_score(actual_peak, daily_peaks)}")
print(f"Counts Actual > HI: {np.sum(actual_peak > daily_hi)}")
print(f"Counts Actual < LO: {np.sum(actual_peak < daily_lo)}")


weekend_mask = (time.dt.weekday == 5) | (time.dt.weekday == 6)
holiday_mask = time.isin(holidays)

plt.style.use("seaborn-v0_8-whitegrid")  # nice default styling
fig, ax = plt.subplots()
plt.scatter(np.abs(np.diff(filtered_temp)), np.abs(residuals)[0:-1])
ax.set_title("Residuals")
ax.set_xlabel("Temperature (\u00b0C)")
ax.set_ylabel("Absolute Residuals")
plt.show()

fig, ax1 = plt.subplots()

ax1.plot(time, np.abs(residuals), color="blue", label="y1")
ax1.set_ylabel("Absolute Residuals (MW)", color="blue")

ax2 = ax1.twinx()
ax2.plot(time, filtered_temp, color="red", label="y2")
ax2.set_ylabel('Temperature (\u00b0C)', color="red")

plt.show()


