import datetime
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import python.colour_dict as colour_dict
from python.public_holidays import get_holidays
from fitting import get_data

# name dictionary for nicer plots:
name_dict = {'rainfall': 'Daily Rainfall',
             "solar_16": "Solar (16 lags)",
             "solar_4": "Solar (4 lags)",
             "pv_capacity": "PV Capacity",
             "temperature": "Temperature",
             "temp_9":"Temperature (9 lags)",
             "cos_336_1": "Fourier Term (Cos, K=1)",
             "sin_336_1": "Fourier Term (Sin, K=1)",
             "temp_1":"Temperature (1 lag)",
             "solar_power": "Solar Irradiance",
             "sar1" : f"Seasonal AR$_1$",
              "ar1" : f"AR$_1$",
             "lag_48*7": "Demand (Lagged, 1 Week)",
             "sma1": f"Seasonal MA$_1$",
             "ma1": f"MA$_1$"}

def get_window_mean(row, data):
    end = pd.to_datetime(row['eval_date'])
    start = end - pd.Timedelta(days=7 * 8)
    try:
        mean = data[row['label']][start:end].mean()
        # catches sin/cos and other near-zero means in a lazy way.
        if np.isnan(mean) or np.abs(mean) < 1e-3:
            return 1.0
        return mean
    except KeyError:
        # nan the AR/MA terms (will plot them a different way)
        return np.nan

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


def plot_impact(data_file, labels_to_plot, save_location, file_name, plt_title=None):

    fig, axes = plt.subplots(len(labels_to_plot), 1, figsize=(14, 3 * len(labels_to_plot)), sharex=True)
    for ax, label in zip(axes, labels_to_plot):
        subset = data_file[data_file['label'] == label].sort_values('eval_date')
        subset['eval_date'] = pd.to_datetime(subset['eval_date'], yearfirst=True)
        ax.plot(subset['eval_date'], subset['impact_coef'], label='coef')
        # fill between using the standard errors converter to 95% CI.
        ax.fill_between(subset['eval_date'],
                        subset['impact_coef'] - 1.96 * subset['impact_std_error'],
                        subset['impact_coef'] + 1.96 * subset['impact_std_error'],
                        alpha=0.2)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_ylabel(label, rotation=90, labelpad=0)

    # clean up the axes
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle(plt_title)
    plt.tight_layout()
    plt.savefig(save_location / file_name, dpi=600)


def data_assessment(df, plotting=False, plt_name="new-plot"):
    df.dropna(how='any', axis=0, inplace=True)
    time = pd.to_datetime(df['eval_date'], yearfirst=True)
    actual_peak_afternoon = (df['peak_actual_afternoon'])
    pred_mean_peak_afternoon = (df['peak_predicted_afternoon_mean'])
    pred_hi_peak_afternoon = (df['peak_predicted_afternoon_hi'])
    pred_low_peak_afternoon = (df['peak_predicted_afternoon_lo'])

    actual_peak_morning = (df['peak_actual_morning'])
    pred_mean_peak_morning = (df['peak_predicted_morning_mean'])
    pred_hi_peak_morning = (df['peak_predicted_morning_hi'])
    pred_low_peak_morning = (df['peak_predicted_morning_lo'])
    actual_peak = np.max([actual_peak_afternoon, actual_peak_morning], axis=0)

    daily_peaks_arg = np.argmax([pred_mean_peak_afternoon, pred_mean_peak_morning], axis=0)

    daily_peaks = np.array([pred_mean_peak_afternoon, pred_mean_peak_morning])
    daily_peaks = daily_peaks[daily_peaks_arg, np.arange(daily_peaks.shape[1])]

    daily_hi = np.array([pred_hi_peak_afternoon, pred_hi_peak_morning])
    daily_hi = daily_hi[daily_peaks_arg, np.arange(daily_hi.shape[1])]

    daily_lo = np.array([pred_low_peak_morning, pred_low_peak_afternoon])
    daily_lo = daily_lo[daily_peaks_arg, np.arange(daily_lo.shape[1])]

    if plotting:
        plot_peaks(actual_peak, daily_hi, daily_lo, daily_peaks,
                   time, plt_name, "Peak Electricity Demand (MW)", plt_title="SARIMAX")

    print(f"RMSE: {root_mean_squared_error(actual_peak, daily_peaks)}")
    print(f"MAPE: {mean_absolute_percentage_error(actual_peak, daily_peaks)}")
    print(f"R2: {r2_score(actual_peak, daily_peaks)}")
    print(f"Counts Actual > HI: {np.sum(actual_peak > daily_hi)}")
    print(f"Counts Actual < LO: {np.sum(actual_peak < daily_lo)}")

def p_value_plots(stats_df, output_lcl, plotting=False):
    heatmap_df = stats_df.pivot(index='eval_date', columns='label', values='p_value')

    # get just the important values p < 0.05
    sig_rate = (heatmap_df < 0.05).mean().sort_values()
    heatmap_df = heatmap_df[sig_rate.index]

    fig, (ax_heat, ax_bar) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [4, 1]})

    # Heatmap — cap at 0.1 so colour difference near zero is visible
    im = ax_heat.imshow(heatmap_df.values.T, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.1)
    ax_heat.set_yticks(range(len(heatmap_df.columns)))
    ax_heat.set_yticklabels(heatmap_df.columns)
    ax_heat.set_xlabel('Rolling Window')
    ax_heat.set_title('P-value Across Rolling Windows')
    plt.colorbar(im, ax=ax_heat, orientation='vertical', label='p-value')

    # bar chart to also make things more obvious?
    ax_bar.bar(range(len(sig_rate)), sig_rate.values, color='steelblue')
    ax_bar.set_xticks(range(len(sig_rate)))
    ax_bar.set_xticklabels(sig_rate.index, rotation=45, ha='right')
    # 80% pass line
    ax_bar.axhline(0.8, color='red', linestyle='--', linewidth=1)
    ax_bar.set_ylabel('% p<0.05')
    ax_bar.set_ylim(0, 1)

    plt.tight_layout()

    if plotting:
        plt.savefig(root_folder / "figures" / (output_lcl+".png"), dpi=600)
    else:
        plt.show()


cwd = Path.cwd()
root_folder = cwd.parent.parent
data_folder = root_folder / "data"
plt_folder = cwd.parent / "figures"

data = get_data(data_folder)
data["date"] = pd.to_datetime(data.index)
data["date"] = data["date"].dt.date
data["pv_capacity"] = data["pv_capacity"]/1000


#exog = pd.read_csv(cwd / 'final_results_window_2026-04-22_with_exog.csv')
no_exog = pd.read_csv(cwd / 'actual_final_results_window_2026-04-24_with_exog_holidays.csv')
#data_assessment(no_exog,plotting=True, plt_name="Final_Window_SARIMAX.png")
stats_df = pd.read_csv(root_folder / "python" / "SARIMAX" / "actual_final_results_window_2026-04-24_with_exog_holidaysstats.csv")

failed_simulations  = no_exog["eval_date"][pd.isna(no_exog["model aicc"])]

# drop the failed simulations
no_exog.dropna(inplace=True, how="any",axis=0)
stats_df = stats_df[~stats_df["eval_date"].isin(failed_simulations)]

stats_df['window_mean'] = stats_df.apply(get_window_mean, axis=1, data=data)
stats_df['impact_coef'] = stats_df['value'] * stats_df['window_mean']
stats_df['impact_std_error'] = stats_df['std_error'] * stats_df['window_mean']

# replace the names of the labels because they are awful:
stats_df["label"].replace(name_dict, inplace=True)
data.rename(columns=name_dict, inplace=True)

# all if you want them.
#labels = stats_df["label"].unique()
#labels = ["Fourier Term (Cos, K=1)",  "Fourier Term (Sin, K=1)", "Demand (Lagged, 1 Week)"]
#labels= ["Temperature", "Temperature (1 lag)", "Temperature (9 lags)"]
#labels = ["Solar Irradiance", "Solar (4 lags)", "Solar (16 lags)"]
labels = ["Daily Rainfall", "PV Capacity", "Holidays", "Weekends"]
file_name = "Other_Impact.png"

#p_value_plots(stats_df, "final_values.png", True)

plot_impact(data_file=stats_df, labels_to_plot=labels, save_location= root_folder/"figures", file_name=file_name)

if False:

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
               time, "SARIMAX_peak_daily_squared", "Peak Electricity Demand (MW)", plt_title="SARIMAX")

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


