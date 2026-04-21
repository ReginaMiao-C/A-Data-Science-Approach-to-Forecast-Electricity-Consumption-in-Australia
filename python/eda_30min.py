import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
import python.colour_dict as cd
from public_holidays import get_holidays





def cross_corr(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return np.correlate(x, y, mode='full')

def determine_lags(data, left, right, max_lags=48, top_n=5):

    """
    Functon to determine the lags needed to best correlate the left and right columns of the data.
    :param data: Dataframe of date
    :param left: name of the "left" column
    :param right: name of the "right" column
    :param max_lags: maximum number of lags to use
    :param top_n: number of lags to return
    :return: list of n integers for the sorted lags.
    """

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

def triplot(dataframe, location_to_save, filler=False, _title=None):
    """
    Function to generate a triplot chart, with Power on the right and Temperature and solar on the left.
    Separate axis for all three so that we can visualise behaviour

    :param dataframe: data to plot
    :param location_to_save: location to save figure (if None, just shows it)
    :param filler: option to plot the mean +/- std for range analysis.
    :return: None.
    """


    if filler:
        mean = dataframe.mean()
        std = dataframe.std()

    fig, ax1 = plt.subplots()
    # First axis (Demand)
    sns.lineplot(data=dataframe, x="hour_of_day", y="total_demand", ax=ax1, color=cd.demand_cols["all"])

    if filler:
        sns.lineplot(data=mean+std, x="time_hours", y="total_demand", ax=ax1, color=cd.demand_cols["all"], alpha=0.3)
        ax = sns.lineplot(data=mean-std, x="time_hours", y="total_demand", ax=ax1, color=cd.demand_cols["all"], alpha=0.3)
        lines = ax.get_lines()
        plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color=cd.demand_cols["all"], alpha=0.1)

    ax1.set_ylabel("Power Demand (MW)", color=cd.demand_cols["all"])
    ax1.set_xlabel("Hour of Day")

    # Second axis (Temp)
    ax2 = ax1.twinx()
    sns.lineplot(data=dataframe, x="hour_of_day", y="temperature", ax=ax2, color=cd.demand_cols["var3"])

    if filler:
        sns.lineplot(data=mean+std, x="time_hours", y="temperature", ax=ax2, color=cd.demand_cols["var3"], alpha=0.3)
        ax = sns.lineplot(data=mean-std, x="time_hours", y="temperature", ax=ax2, color=cd.demand_cols["var3"], alpha=0.3)
        lines = ax.get_lines()
        plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color=cd.demand_cols["var3"], alpha=0.1)

    ax2.set_ylabel(cd.var_dict_peak["temperature"], color=cd.demand_cols["var3"])

    # Third axis (Solar Power)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    sns.lineplot(data=dataframe, x="hour_of_day", y="solar_power", ax=ax3, color=cd.demand_cols["var4"])

    if filler:
        sns.lineplot(data=mean+std, x="time_hours", y="solar_power", ax=ax3, color=cd.demand_cols["var4"], alpha=0.3)
        ax = sns.lineplot(data=mean-std, x="time_hours", y="solar_power", ax=ax3, color=cd.demand_cols["var4"], alpha=0.3)
        lines = ax.get_lines()
        plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color=cd.demand_cols["var4"], alpha=0.1)

    ax3.set_ylabel(cd.var_dict_peak["solar_power"], color=cd.demand_cols["var4"])

    if _title is not None:
        plt.title(_title, color="black")

    plt.xticks(rotation=45)
    plt.tight_layout()


    if location_to_save is not None:
        plt.savefig(location_to_save)
    else:
        plt.show()


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"

# stack of date/time varibles need to analysed so just pool them into one big blob
data = pd.read_csv(data_folder / "all_data_30min.csv")
data["datetime"] = pd.to_datetime(data["datetime"], yearfirst=True)
data.index = data["datetime"]
data["date"] = data["datetime"].dt.date
data["day_of_the_year"] = data["datetime"].dt.dayofyear
data["hour_of_day"] = data["datetime"].dt.hour + data["datetime"].dt.minute/60
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day_name()
data["year"] = data["datetime"].dt.year
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#data.drop("datetime", axis=1, inplace=True)

data = data.iloc[100:, :]

day_group = data.groupby("day")

plt.figure(figsize=(12, 8))

day_colors = {
    'Monday': 'tab:blue',
    'Tuesday': 'tab:orange',
    'Wednesday': 'tab:green',
    'Thursday': 'tab:red',
    'Friday': 'tab:purple',
    'Saturday': 'tab:brown',
    'Sunday': 'tab:pink'
}


data["hour_of_week"] = data["datetime"].dt.hour + data["datetime"].dt.minute / 60 + data[
        "datetime"].dt.dayofweek * 24

fig, ax1 = plt.subplots()
data.groupby("hour_of_week")["total_demand"].mean().plot(color=cd.demand_cols["all"])
(data.groupby("hour_of_week")["total_demand"].std() + data.groupby("hour_of_week")["total_demand"].mean()).plot(
    alpha=0.1, color=cd.demand_cols["all"])
(data.groupby("hour_of_week")["total_demand"].mean()- data.groupby("hour_of_week")["total_demand"].std()).plot(
    alpha=0.1, color=cd.demand_cols["all"])
lines = ax1.get_lines()
plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color=cd.demand_cols["all"], alpha=0.1)
ax1.set_xlabel("Hour of the Week")
ax1.set_ylabel("Power Demand (MW)")
plt.savefig(root_folder / "figures" / "daily_mean_week_std.png")



if False:

    plt.figure(figsize=(10, 6))
    for day in days_order:
        temp_df = day_group.get_group(day)
        temp_ = temp_df.groupby("hour_of_day")["total_demand"].mean()
        plt.plot(temp_.index, temp_.values, color=day_colors[day], marker='o', label=day)

    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand")
    plt.tight_layout()
    plt.savefig(root_folder / "figures" / "day_by_day_stacked.png")


    holidays = []

    for year in range(data["year"].min(), data["year"].max()):
        holidays.extend(get_holidays(year))

    holiday_peak = []
    not_holiday_peak = []

    for holiday in holidays:
        holiday_peak.append(data[data["date"] == holiday]["total_demand"].max())

        not_holiday_peak.append(
            0.5 * (data[data["date"] == holiday + timedelta(days=7)]["total_demand"].max() +
                   data[data["date"] == holiday - timedelta(days=7)]["total_demand"].max())
        )

    holiday_data = pd.DataFrame([holiday_peak, not_holiday_peak], dtype=float).T
    holiday_data.columns = ["holiday_peak", "not_holiday_peak"]
    holiday_data.index = holidays
    # the data cleaning removed New Years 2010
    holiday_data.dropna(inplace=True, how='all', axis=0)
    holiday_data.plot()
    plt.savefig(root_folder / "figures" / "holiday_vs_non_holiday.png")

    print(determine_lags(winter_data, "total_demand", "temperature"))
    print(determine_lags(winter_data, "total_demand", "solar_power"))

    print(determine_lags(summer_data, "total_demand", "temperature"))
    print(determine_lags(summer_data, "total_demand", "solar_power"))

    print(determine_lags(data, "total_demand", "temperature", 12))
    print(determine_lags(data, "total_demand", "solar_power", 12))

    sns.lineplot(data=data, x="day_of_the_year", y="total_demand", ax=plt.gca())
    plt.savefig(root_folder / "figures" / "daily_variation.png")

    data["hour_of_week"] = data["datetime"].dt.hour + data["datetime"].dt.minute / 60 + data[
        "datetime"].dt.dayofweek * 24

    fig, ax1 = plt.subplots()
    data.groupby("hour_of_week").mean()["total_demand"].plot()
    (data.groupby("hour_of_week").std()["total_demand"] + data.groupby("hour_of_week").mean()["total_demand"]).plot(
        alpha=0.1)
    (data.groupby("hour_of_week").mean()["total_demand"] - data.groupby("hour_of_week").std()["total_demand"]).plot(
        alpha=0.1)
    lines = ax1.get_lines()
    plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color='blue', alpha=0.1)

    plt.savefig(root_folder / "figures" / "daily_mean_week_std.png")

    winter_solstice = 172
    spring_equinox = 264
    summer_solstice = 355
    autumn_equinox = 81

    winter_data = data.drop(columns=["day", "date"], inplace=False)[
        (winter_solstice - 14 < data["day_of_the_year"]) & (winter_solstice + 14 > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()
    spring_data = data.drop(columns=["day", "date"], inplace=False)[
        (spring_equinox - 14 < data["day_of_the_year"]) & (spring_equinox + 14 > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()
    summer_data = data.drop(columns=["day", "date"], inplace=False)[
        (summer_solstice - 14 < data["day_of_the_year"]) & (summer_solstice + 14 > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()
    autumn_data = data.drop(columns=["day", "date"], inplace=False)[
        (autumn_equinox - 14 < data["day_of_the_year"]) & (autumn_equinox + 14 > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()

    mean_daily = data.drop(columns=["day", "date"], inplace=False).groupby("hour_of_day").mean()

    triplot(winter_data, root_folder / "figures" / "winter_solstice.png", _title="Winter Solstice")
    triplot(summer_data, root_folder / "figures" / "summer_solstice.png", _title="Summer Solstice")
    triplot(spring_data, root_folder / "figures" / "spring_equinox.png", _title="Spring Equinox")
    triplot(autumn_data, root_folder / "figures" / "autumn_equinox.png", _title="Autumn Equinox")
    triplot(mean_daily, root_folder / "figures" / "mean_daily.png", _title="Mean Daily")

    # pivot table for the plotting
    heatmap_data = data.pivot_table(index='day', columns='hour', values='total_demand', aggfunc='mean')

    # make sure the days are ordered correctly.
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='viridis')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.savefig(root_folder / "figures" / "heatmap_of_demand.png")

    years = np.sort(data["year"].unique())
    # drop the first and last year due to incomplete data
    years = years[1:-2]

    energy_by_year = []

    for year in years:
        demand = data[data["year"] == year]["total_demand"]
        index = data[data["year"] == year].index
        # convert time interval in hours from first:
        time = (index - index[0]).total_seconds()/(60*60)

        # integrate to find the total energy and convert from MWhr to Petajoules
        total_demand = np.trapezoid(demand.values, time)/10**3/277.8
        energy_by_year.append((year, total_demand))

    energy_by_year = np.array(energy_by_year)

    plt.plot(energy_by_year[:,0], energy_by_year[:,1])
    plt.xlabel("Year")
    plt.ylabel("Total Energy (Petajoules)")
    #plt.ylim([0, 300])
    plt.savefig(root_folder / "figures" / "yearly_power_usage_free_axis.png")

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
