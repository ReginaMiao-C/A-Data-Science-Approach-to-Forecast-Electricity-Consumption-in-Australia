import numpy as np
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import python.colour_dict as cd
from public_holidays import get_holidays
from colour_dict import day_colors


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

    # normalise the data, to assist in the correlation finding
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    corr = np.correlate(x, y, mode='full')
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

    ax1.set_ylabel("Electricity Demand (MW)", color=cd.demand_cols["all"])
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


def seasonal_triplots(data, sve_lction,  days_buffer):
    """
    Plots of the seasonal data based on the equinox/solstice, triplots of the power, temp and solar irradiance.

    :param data: dataframe of data
    :param days_buffer: number of days either side of the critcal day to assess
    :param sve_lction: folder to save the images in
    :return: None. Saves data to the folder

    """

    # dates for the different seasonal points from January 1.
    autumn_equinox = 81
    winter_solstice = 172
    spring_equinox = 264
    summer_solstice = 355

    winter_data = data.drop(columns=["day", "date"], inplace=False)[
        (winter_solstice - days_buffer < data["day_of_the_year"]) & (winter_solstice + days_buffer > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()
    spring_data = data.drop(columns=["day", "date"], inplace=False)[
        (spring_equinox - days_buffer < data["day_of_the_year"]) & (spring_equinox + days_buffer > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()
    summer_data = data.drop(columns=["day", "date"], inplace=False)[
        (summer_solstice - days_buffer < data["day_of_the_year"]) & (summer_solstice + days_buffer > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()
    autumn_data = data.drop(columns=["day", "date"], inplace=False)[
        (autumn_equinox - days_buffer < data["day_of_the_year"]) & (autumn_equinox + days_buffer > data["day_of_the_year"])].groupby(
        "hour_of_day").mean()

    # print some information about the lags:
    print(f"Winter lags vs Temperature: {determine_lags(winter_data, "total_demand", "temperature")}")
    print(f"Winter lags vs Solar Irradiance: {determine_lags(winter_data, "total_demand", "solar_power")}")

    print(f"Summer lags vs Temperature: {determine_lags(summer_data, "total_demand", "temperature")}")
    print(f"Summer lags vs Solar Irradiance: {determine_lags(summer_data, "total_demand", "solar_power")}")

    mean_daily = data.drop(columns=["day", "date"], inplace=False).groupby("hour_of_day").mean()

    triplot(winter_data, sve_lction /  "winter_solstice.png", _title="Winter Solstice")
    triplot(summer_data, sve_lction / "summer_solstice.png", _title="Summer Solstice")
    triplot(spring_data, sve_lction / "spring_equinox.png", _title="Spring Equinox")
    triplot(autumn_data, sve_lction / "autumn_equinox.png", _title="Autumn Equinox")
    triplot(mean_daily, sve_lction / "mean_daily.png", _title="Mean Daily")


def hour_of_the_week(data, sve_lction):
    """
    Plot of the data over a week averaged over all data with CI bands directly from Seaborn
    :param data: dataframe of data (needs to contain the parameter: hour_of_week)
    :param sve_lction: folder to save the images in
    :return: None. Saves images to the folder
    """

    fig, ax1 = plt.subplots()
    data.groupby("hour_of_week")["total_demand"].mean().plot(color=cd.demand_cols["all"])
    (data.groupby("hour_of_week")["total_demand"].std() + data.groupby("hour_of_week")["total_demand"].mean()).plot(
        alpha=0.1, color=cd.demand_cols["all"])
    (data.groupby("hour_of_week")["total_demand"].mean() - data.groupby("hour_of_week")["total_demand"].std()).plot(
        alpha=0.1, color=cd.demand_cols["all"])
    lines = ax1.get_lines()
    plt.fill_between(lines[0].get_xdata(), lines[1].get_ydata(), lines[2].get_ydata(), color=cd.demand_cols["all"],
                     alpha=0.1)
    ax1.set_xlabel("Hour of the Week")
    ax1.set_ylabel("Electricity Demand (MW)")
    plt.tight_layout()
    plt.savefig(sve_lction/ "daily_mean_week_std.png")

    # days in the proper order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_group = data.groupby("day")

    plt.figure(figsize=(10, 6))
    for day in days_order:
        temp_df = day_group.get_group(day)
        temp_ = temp_df.groupby("hour_of_day")["total_demand"].mean()
        plt.plot(temp_.index, temp_.values, color=day_colors[day], marker='o', label=day)

    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand")
    plt.tight_layout()
    plt.savefig(sve_lction / "day_by_day_stacked.png")


def effect_of_holidays(data, sve_lcatoin):
    """
    plots the average effect of holidays vs non-holiday on the peak demand,
    :param data: dataframe of data
    :param sve_lcatoin: folder to save the images in
    :return: None, images are saved to file
    """

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
    plt.ylabel("Average Electricity Demand")
    plt.savefig(sve_lcatoin / "holiday_vs_non_holiday.png")


def yearly_energy_usage(data, sve_lcatoin, free_axis=False):
    """
    Integrates the demand data to get the total usage by year.
    :param data: Dataframe of data
    :param sve_lcatoin: folder to save the images in
    :param free_axis: Whether to plot the data with 0 in the y-axis, or not (good for seeing local trends vs global trends)
    :return: None, saves data to folder.
    """

    energy_by_year = []
    for year in data["year"].unique():
        demand = data[data["year"] == year]["total_demand"]
        index = data[data["year"] == year].index
        # convert time interval in hours from first:
        time = (index - index[0]).total_seconds() / (60 * 60)

        # integrate to find the total energy and convert from MWhr to Petajoules
        total_demand = np.trapezoid(demand.values, time) / 10 ** 3 / 277.8
        energy_by_year.append((year, total_demand))

    energy_by_year = np.array(energy_by_year)

    plt.plot(energy_by_year[:, 0], energy_by_year[:, 1])
    plt.xlabel("Year")
    plt.ylabel("Total Energy (Petajoules)")
    plt.tight_layout()

    if not free_axis:
        plt.ylim([0,None])

    plt.savefig(sve_lcatoin/"yearly_power_usage_free_axis.png")


def main():
    """
    Main function to add additional information in the data array and dispatch for plotting.
    :return: None, plots are saved to folder.
    """


    cwd = Path.cwd()
    root_folder = cwd.parent
    data_folder = root_folder / "data"
    sve_folder = root_folder / "figures"

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
    data["hour_of_week"] = data["datetime"].dt.hour + data["datetime"].dt.minute / 60 + data["datetime"].dt.dayofweek * 24

    # get rid of nans.
    data.dropna(inplace=True, how='any', axis=0)

    seasonal_triplots(data, sve_folder, days_buffer=14)
    hour_of_the_week(data, sve_folder)
    yearly_energy_usage(data, sve_folder, free_axis=True)
    effect_of_holidays(data, sve_folder)


if __name__ == "__main__":
    main()