from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import python.colour_dict as colour_dict
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
    """
        Helper function to calculate mean of a dataframe rows evaluation date and label,
        returns 1.0 for near zero mean and np.nan for key errors
        :param row: pd.Series row of data for the analysis
        :param data: the full dataset for the analysis of the mean
        :return: float/np.nan the mean for the given row label and evaluation date.
    """

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
    """
    Plot the peaks of the prediction and the CI interval along with scattered values for the actual peak demand data.

    :param actual: pd.Series containing the actual demand data
    :param hi: pd.Series containing the predicted upper CI demand data
    :param low: pd.Series containing the predicted lower CI demand data
    :param mean: pd.Series containing the mean predicted demand data
    :param x: pd.Series containing the time series data
    :param name: save name of the plot
    :param axis_name: name of the axis
    :param plt_title: title of the plot (optional, defaults to None)
    :return: None, saves plots to the figures folder.
      """


    cwd = Path.cwd()

    # nice default styling
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(x, low, hi, color="tab:blue", alpha=0.2, label="Prediction interval")
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


def plot_impact(data_file, labels_to_plot, save_location, plt_title=None):
    """
    plots the coefficient values and the range based on the reported 95% CI for the requested labels based on the
    standard errors.

   :param data_file: data file containing the coefficient values and the standard errors
   :param labels_to_plot: list of labels to plot
   :param save_location: folder to save the plot in
   :param plt_title: title of the plot (optional, defaults to None)
   :return: None, saves plots to the figures folder.
    """

    fig, axes = plt.subplots(len(labels_to_plot), 1, figsize=(14, 3 * len(labels_to_plot)), sharex=True)
    for ax, label in zip(axes, labels_to_plot):
        subset = data_file[data_file['label'] == label].sort_values('eval_date')
        subset['eval_date'] = pd.to_datetime(subset['eval_date'], yearfirst=True)
        ax.plot(subset['eval_date'], subset['impact_coef'], label='coef')
        # fill between using the standard errors converter to 95% CI. 1.96 ~ qnorm(0.975, mean=0, sd=1)
        ax.fill_between(subset['eval_date'],
                        subset['impact_coef'] - 1.96 * subset['impact_std_error'],
                        subset['impact_coef'] + 1.96 * subset['impact_std_error'],
                        alpha=0.2)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_ylabel(label, rotation=90, labelpad=0, fontsize=14)

    # clean up the axes
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=14)

    plt.suptitle(plt_title)
    plt.tight_layout()
    plt.savefig(save_location, dpi=600)


def data_assessment(data, plotting=False, plt_name="new-plot"):
    """
    Conducts an assessment of the results, mostly just printed to the console with the option to generate the plots
      :param data: dataframe of data
      :param plotting: boolean for if the plotting of the peaks is to be done
      :param plt_name: name of save file for the plots
      :return: None, saves plots to the figures folder, prints stats to the console.
    """

    data.dropna(how='any', axis=0, inplace=True)
    time = pd.to_datetime(data['eval_date'], yearfirst=True)
    actual_peak_afternoon = (data['peak_actual_afternoon'])
    pred_mean_peak_afternoon = (data['peak_predicted_afternoon_mean'])
    pred_hi_peak_afternoon = (data['peak_predicted_afternoon_hi'])
    pred_low_peak_afternoon = (data['peak_predicted_afternoon_lo'])

    actual_peak_morning = (data['peak_actual_morning'])
    pred_mean_peak_morning = (data['peak_predicted_morning_mean'])
    pred_hi_peak_morning = (data['peak_predicted_morning_hi'])
    pred_low_peak_morning = (data['peak_predicted_morning_lo'])
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
    print(f"MSE: {mean_squared_error(actual_peak, daily_peaks)}")
    print(f"MAE: {mean_absolute_error(actual_peak, daily_peaks)}")
    print(f"MAPE: {mean_absolute_percentage_error(actual_peak, daily_peaks)}")
    print(f"R2: {r2_score(actual_peak, daily_peaks)}")
    print(f"Counts Actual > HI: {np.sum(actual_peak > daily_hi)}")
    print(f"Counts Actual < LO: {np.sum(actual_peak < daily_lo)}")



def p_value_plots(stats_df, output_lcl=None):
    """
    Heatmap of the exogenous and AR/MA p-values along with bar plot of the number of days when the value is significant.

    :param stats_df: dataframe of statistical data from the analysis
    :param output_lcl: location to save the plot to or just print if None
    :return:
    """

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

    if output_lcl is not None:
        plt.savefig(output_lcl, dpi=600)
    else:
        plt.show()


def main():
    """
    Main function to run the assessment of the results.
    :param: None (all controls are inside this function)
    :return: None, files are saved and data is printed to the console.
    """

    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    plt_folder = root_folder / "figures"

    # need the data again for the plotting of the coefficients (for the mean values)
    data = get_data(data_folder)
    data["date"] = pd.to_datetime(data.index)
    data["date"] = data["date"].dt.date
    data["pv_capacity"] = data["pv_capacity"] / 1000

    # name of the datafile to analyse
    data_file = "daily_data_with_exog2026-04-24"
    # switch for whether the data contains the exogenous variables, and you want to plot the coefficients over the window.
    exogenous_ananlysis = True
    # plot switch
    save_plots = True
    plt_name = "Final_Window_SARIMAX.png"

    results_data = pd.read_csv(cwd /"data"/ (data_file + ".csv"))
    stats_df = pd.read_csv(cwd /"data"/ (data_file + "_stats.csv"))

    # drop any failed simulations
    failed_simulations = results_data["eval_date"][pd.isna(results_data["model aicc"])]
    results_data.dropna(inplace=True, how="any", axis=0)
    stats_df = stats_df[~stats_df["eval_date"].isin(failed_simulations)]

    data_assessment(results_data,plotting=save_plots, plt_name=plt_name)

    # analysis of the coefficients is only done for the exogenous variables.
    if exogenous_ananlysis:
        stats_df['window_mean'] = stats_df.apply(get_window_mean, axis=1, data=data)
        stats_df['impact_coef'] = stats_df['value'] * stats_df['window_mean']
        stats_df['impact_std_error'] = stats_df['std_error'] * stats_df['window_mean']

        # replace the names of the labels because they are awful:
        stats_df["label"] = stats_df["label"].replace(name_dict)
        data.rename(columns=name_dict, inplace=True)

        p_value_plots(stats_df, plt_folder/"final_values.png")

        # all if you want them, or just a subset of variables as it looks better on the plots.
        # labels = stats_df["label"].unique()
        labels = [("Weekly_Impact", ["Fourier Term (Cos, K=1)",  "Fourier Term (Sin, K=1)", "Demand (Lagged, 1 Week)"]),
                  ("Temp_Impact", ["Temperature", "Temperature (1 lag)", "Temperature (9 lags)"]),
                  ("Solar_impact",["Solar Irradiance", "Solar (4 lags)", "Solar (16 lags)"]),
                  ("Date_Impact", ["Hour", "Day of the Week", "Month"]),
                  ("Other_impact", ["Daily Rainfall", "PV Capacity", "Holidays", "Weekends"])]

        for (name, label) in labels:
            plot_impact(data_file=stats_df, labels_to_plot=label, save_location=plt_folder / (name + ".png"))




if __name__ == "__main__":
    main()


