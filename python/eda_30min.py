import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys
import matplotlib as mpl


def cross_corr(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return np.correlate(x, y, mode='full')

def determine_lags(data, left, right, max_lags=48, top_n=5):
    x = data[left].values
    y = data[right].values

    corr = cross_corr(x, y)
    lags = np.arange(-len(x) + 1, len(x))

    # Keep only positive lags and those less than the maximum
    mask = (lags >= 0) & (lags <= max_lags)

    lags_limited = lags[mask]
    corr_limited = corr[mask]

    # Rank by absolute correlation
    idx = np.argsort(np.abs(corr_limited))[::-1]

    return lags_limited[idx[:top_n]]




def triplot(dataframe, location_to_save):
    fig, ax1 = plt.subplots()
    # First axis (Demand)
    sns.lineplot(data=dataframe, x="hour_of_day", y="total_demand", ax=ax1, color="blue")
    # sns.lineplot(data=mean+std, x="time_hours", y="total_demand", ax=ax1, color="blue", alpha=0.3)
    # ax = sns.lineplot(data=mean-std, x="time_hours", y="total_demand", ax=ax1, color="blue", alpha=0.3)
    # lines = ax.get_lines()
    # plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color='blue', alpha=0.1)

    ax1.set_ylabel("Power", color="blue")

    # Second axis (Temp)
    ax2 = ax1.twinx()
    sns.lineplot(data=dataframe, x="hour_of_day", y="temperature", ax=ax2, color="red")
    # sns.lineplot(data=mean+std, x="time_hours", y="temperature", ax=ax2, color="red", alpha=0.3)
    # ax = sns.lineplot(data=mean-std, x="time_hours", y="temperature", ax=ax2, color="red", alpha=0.3)
    # lines = ax.get_lines()
    # plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color='red', alpha=0.1)
    ax2.set_ylabel("Temperature", color="red")

    # Third axis (Solar Power)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    sns.lineplot(data=dataframe, x="hour_of_day", y="solar_power", ax=ax3, color="orange")
    # sns.lineplot(data=mean+std, x="time_hours", y="solar_power", ax=ax3, color="orange", alpha=0.3)
    # ax = sns.lineplot(data=mean-std, x="time_hours", y="solar_power", ax=ax3, color="orange", alpha=0.3)
    # lines = ax.get_lines()
    # plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color='orange', alpha=0.1)
    ax3.set_ylabel("Solar Power", color="orange")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.tight_layout()


    if location_to_save is not None:
        plt.savefig(location_to_save)
    else:
        plt.show()




cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"


data = pd.read_csv(data_folder / "all_data_30min.csv")
data["datetime"] = pd.to_datetime(data["datetime"], yearfirst=True)
data.index = data["datetime"]
data["day_of_the_year"] = data["datetime"].dt.dayofyear
data["hour_of_day"] = data["datetime"].dt.hour + data["datetime"].dt.minute/60
#data.drop("datetime", axis=1, inplace=True)

data = data.iloc[100:, :]

winter_solstice = [21, 6]
summer_solstice = [21, 12]

winter_data = data[data["day_of_the_year"] == 172].groupby("hour_of_day").mean()
summer_data = data[data["day_of_the_year"] == 355].groupby("hour_of_day").mean()

#triplot(winter_data, root_folder / "figures" / "winter_solstice.png")
#triplot(summer_data, root_folder / "figures" / "summer_solstice.png")

print(determine_lags(winter_data, "total_demand", "temperature"))
print(determine_lags(winter_data, "total_demand", "solar_power"))

print(determine_lags(summer_data, "total_demand", "temperature"))
print(determine_lags(summer_data, "total_demand", "solar_power"))

print(determine_lags(data, "total_demand", "temperature", 12))
print(determine_lags(data, "total_demand", "solar_power", 12))



if False:    # Plot
    x = data.index.weekday
    y = data["total_demand"]

    # Fit quadratic model
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs

    # Predictions
    y_pred = np.polyval(coeffs, x)

    # R^2
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Helper to format signs nicely
    def fmt(val, precision=2):
        return f"{abs(val):.{precision}f}"

    sign_b = "+" if b >= 0 else "-"
    sign_c = "+" if c >= 0 else "-"

    # LaTeX-style equation
    eq_text = (
        rf"$y = {a:.2f}x^2 {sign_b} {fmt(b)}x {sign_c} {fmt(c)}$" "\n"
        rf"$R^2 = {r2:.3f}$"
    )

    data["date"] = data["datetime"].dt.date
    clean_data = data.drop(columns=["datetime"])

    maxed = data.groupby("date").max()

    # Extract day of year
    maxed["day_of_year"] = maxed["datetime"].dt.dayofyear
    norm = mpl.colors.Normalize(vmin=1, vmax=366)


    # Scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=maxed,
        x="temperature",
        y="total_demand",
        hue="day_of_year",
        palette="twilight",   # good continuous colormap
        legend=False ,       # optional (can get messy with many days)
        hue_norm=norm,
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Power (W)")
    plt.title("Temperature vs Power")
    plt.tight_layout()
    plt.savefig(root_folder / "figures" / "max_daily_temp_vs_power.png")



    # Scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x="solar_power",
        y="total_demand",
       # hue="day_of_year",
        palette="twilight",   # good continuous colormap
        legend=False ,       # optional (can get messy with many days)
       # hue_norm=norm,
    )

    plt.xlabel("Solar Power (W.m^2)")
    plt.ylabel("Power (W)")
    plt.title("Temperature vs Power")
    plt.tight_layout()
    plt.savefig(root_folder / "figures" / "max_daily_power_vs_power.png")

    data["time_hours"] = (
        data["datetime"].dt.hour +
        data["datetime"].dt.minute / 60
    )


    data.drop(columns=["datetime", "date"], inplace=True)
    grouped_data = data.groupby("time_hours")
    mean = grouped_data.mean()
    std = grouped_data.std()
    n = grouped_data.count()

    ci = 1.96 * std / np.sqrt(n)


    plt.figure(figsize=(10, 5))


    # Mean line
    plt.plot(mean.index, mean["total_demand"], color="blue")

    # Confidence interval
    plt.fill_between(
        mean.index,
        mean["total_demand"] - std["total_demand"],
        mean["total_demand"] + std["total_demand"],
        color="blue",
        alpha=0.3
    )


    plt.show()
    fig, ax1 = plt.subplots()
    # First axis (Demand)
    sns.lineplot(data=mean, x="time_hours", y="total_demand", ax=ax1, color="blue")
    #sns.lineplot(data=mean+std, x="time_hours", y="total_demand", ax=ax1, color="blue", alpha=0.3)
    #ax = sns.lineplot(data=mean-std, x="time_hours", y="total_demand", ax=ax1, color="blue", alpha=0.3)
    #lines = ax.get_lines()
    #plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color='blue', alpha=0.1)

    ax1.set_ylabel("Power", color="blue")

    # Second axis (Temp)
    ax2 = ax1.twinx()
    sns.lineplot(data=mean, x="time_hours", y="temperature", ax=ax2, color="red")
    #sns.lineplot(data=mean+std, x="time_hours", y="temperature", ax=ax2, color="red", alpha=0.3)
    #ax = sns.lineplot(data=mean-std, x="time_hours", y="temperature", ax=ax2, color="red", alpha=0.3)
    #lines = ax.get_lines()
    #plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color='red', alpha=0.1)
    ax2.set_ylabel("Temperature", color="red")

    # Third axis (Solar Power)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    sns.lineplot(data=mean, x="time_hours", y="solar_power", ax=ax3, color="orange")
    #sns.lineplot(data=mean+std, x="time_hours", y="solar_power", ax=ax3, color="orange", alpha=0.3)
    #ax = sns.lineplot(data=mean-std, x="time_hours", y="solar_power", ax=ax3, color="orange", alpha=0.3)
    #lines = ax.get_lines()
    #plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color='orange', alpha=0.1)
    ax3.set_ylabel("Solar Power", color="orange")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(root_folder / "figures" / "mean_daily.png")

if False:




    corr = cross_corr(data["temperature"], data["total_demand"])
    lags = np.arange(-len(data)+1, len(data))

    max_lag = 48
    mask = (lags >= -max_lag) & (lags <= max_lag)

    lags_limited = lags[mask]
    corr_limited = corr[mask]

    best_lag = lags_limited[np.argmax(corr_limited)]

    print(f"Best lag: {best_lag} steps (~{best_lag*0.5} hours)")

    corr = cross_corr(data["solar_power"], data["total_demand"])
    lags = np.arange(-len(data)+1, len(data))

    max_lag = 48
    mask = (lags >= -max_lag) & (lags <= max_lag)

    lags_limited = lags[mask]
    corr_limited = corr[mask]

    best_lag = lags_limited[np.argmax(corr_limited)]

    print(f"Best lag: {best_lag} steps (~{best_lag*0.5} hours)")
