import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys
from colour_dict import demand_cols as vc
from colour_dict import var_dict_total as total_labels
from colour_dict import var_dict_peak as peak_labels

def scatterplot_matrix(df, img_folder, img_name, colour):
    """
    scatterplot matrix of all numerical variables in dataframe
    """
    var_labels = ['Rainfall', 'PV Capacity', 'Temperature', 'Solar Irradiance', 'Total Power']
    p = sns.pairplot(df, plot_kws={'alpha': 0.6, 's': 10}, hue='season', palette='Spectral')
    p.x_vars = var_labels
    p.y_vars = var_labels
    p._add_axis_labels()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(img_folder / img_name)
    plt.close()


def corr_matrix (df, peak_df, img_folder):
    """
    correlation matrix of all numerical variables in dataframe
    """
    var_labels = ['Rainfall', 'PV Capacity', 'Temperature', 'Solar Irradiance', 'Total Power']
    
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    for i in range(4):
        row = i // 2
        col = i % 2
        current_season = peak_df['season'].unique()[i]
        season_peak_df = peak_df[peak_df['season']==current_season]
        season_cm = season_peak_df.corr(numeric_only=True)
        sns.heatmap(season_cm, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axs[row, col], cbar=False, xticklabels=var_labels, yticklabels=var_labels)
        axs[row, col].set_title(current_season)
        if row == 0:
            axs[row, col].set(xticklabels=[])
        if col == 1:
            axs[row, col].set(yticklabels=[])
    plt.suptitle('Seasonal Correlation Between Variables')
    plt.tight_layout()
    plt.savefig(img_folder / 'corr_matrices_season_peak')
    plt.close()

    cm = df.corr(numeric_only=True)
    peak_cm = peak_df.corr(numeric_only=True)
    fig, axs = plt.subplots(1, 2, figsize=(11,6))
    sns.heatmap(cm, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axs[0], cbar=False, xticklabels=var_labels, yticklabels=var_labels)
    sns.heatmap(peak_cm, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axs[1], cbar=False, xticklabels=var_labels)
    axs[0].set_title('Electricity Demand')
    axs[1].set_title('Peak Electricity Demand')
    axs[1].tick_params(axis='y', labelleft=False)
    axs[0].tick_params(axis='y', rotation=0)
    plt.suptitle('Correlation Between Variables')
    plt.tight_layout()
    plt.savefig(img_folder / 'corr_matrices')
    plt.close()


def var_distributions(df, peak_df, img_folder):
    """
    compare variable distributions for total and peak demand
    """
    # violin plot matrix
    fig, axs = plt.subplots(2, 5, figsize=(12,6))
    col_list = df.columns.tolist()[1:6]
    for col in col_list:
        col_idx = col_list.index(col)
        sns.violinplot(df[col], ax=axs[0][col_idx], color=vc['all'])
        sns.violinplot(peak_df[col], ax=axs[1][col_idx], color=vc['all'])
        ymin, ymax = axs[0][col_idx].get_ylim()
        axs[1][col_idx].set_ylim(ymin, ymax)
        axs[0][col_idx].set_title(col)
        axs[0][col_idx].set_ylabel('')
        axs[1][col_idx].set_ylabel('')
    fig.text(0.01, 0.68, 'Electricity Demand', va='center', rotation=90, fontsize=12)
    fig.text(0.01, 0.23, 'Peak Electricity Demand', va='center', rotation=90, fontsize=12)
    plt.suptitle('Variable Distributions')
    plt.tight_layout()
    fig.subplots_adjust(left=0.06) 
    plt.savefig(img_folder / 'var_violinplots')
    plt.close()

    # peak demand violin plot by season
    fig, axs = plt.subplots(5, 1, figsize=(6,12))
    for col in col_list:
        col_idx = col_list.index(col)
        sns.violinplot(peak_df, y=col, ax=axs[col_idx], color=vc['all'], hue='season', palette='Spectral')
        if col_idx != 4:
            axs[col_idx].get_legend().remove()
    plt.suptitle('Peak Demand Variable Distributions')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.035) 
    axs[4].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    plt.savefig(img_folder / 'peak_season_violinplots')
    plt.close()


    # histograms of numerical variables
    for col in col_list:
        fig, axs = plt.subplots(1, 2, figsize=(8,5))
        sns.histplot(df, x=col, ax=axs[0], color=vc['all'], bins=50)
        sns.histplot(peak_df, x=col, ax=axs[1], color=vc['peak'], bins=50)
        axs[0].grid(axis='y', linestyle='--', color='lightgrey')
        axs[1].grid(axis='y', linestyle='--', color='lightgrey')
        axs[0].set_axisbelow(True)
        axs[1].set_axisbelow(True)
        xmin, xmax = axs[0].get_xlim()
        axs[1].set_xlim(xmin, xmax)
        axs[1].set_ylabel('')
        axs[0].set_ylabel('Count')
        axs[0].set_xlabel(total_labels[col])
        axs[1].set_xlabel(total_labels[col])
        axs[0].set_title('Total Power')
        axs[1].set_title('Peak Electricity Demand')
        img_name = col + '_histograms'
        plt.savefig(img_folder / img_name)
        plt.close()

    for col in col_list:
        fig, axs = plt.subplots(1, 2, figsize=(6,5), layout='constrained')
        sns.boxplot(df, y=col, ax=axs[0], color=vc['all'])
        sns.boxplot(peak_df, y=col, ax=axs[1], color=vc['peak'])
        axs[0].grid(axis='y', linestyle='--', color='lightgrey')
        axs[1].grid(axis='y', linestyle='--', color='lightgrey')
        ymin, ymax = axs[0].get_ylim()
        axs[1].set_ylim(ymin, ymax)
        axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)
        axs[0].set_ylabel(total_labels[col])
        axs[1].set_ylabel('')
        axs[0].set_title('Electricity Demand')
        axs[1].set_title('Peak Electricity Demand')
        img_name = col + '_boxplot'
        plt.savefig(img_folder / img_name)
        plt.close()

    # histogram showing time of peak demand
    peak_df['hour_float'] = peak_df['datetime'].dt.hour + peak_df['datetime'].dt.minute / 60
    sns.histplot(peak_df, x='hour_float', bins=48, hue='season', multiple='stack', palette='Spectral')
    plt.xticks(range(0, 24, 1))  # show every 2 hours
    plt.xlabel('Hour')
    plt.title('Time Frequency of Peak Electricity Demand')
    plt.savefig(img_folder / 'peak_demand_time_histogram_seasons')
    plt.close()


def demand_time(df, img_folder):
    """
    plot all demand measurements over time
    """
    jan_1 = df[(df['date'].dt.month == 1) & (df['date'].dt.day == 1)]['date']
    plt.figure(figsize=(12, 5))
    sns.lineplot(df, x='date', y='total_demand', lw=1, errorbar=None, color='palevioletred')
    for l in jan_1:
        plt.axvline(x=l, color='lightgrey', linestyle='--', lw=0.5, zorder=1)
    plt.title('Peak Electricity Demand')
    plt.savefig(img_folder / 'peak_demand_time')
    plt.close()

def demand_time_all(df, peak_df, img_folder):
    """
    plot all demand measurements over time
    """
    jan_1 = df[(df['date'].dt.month == 1) & (df['date'].dt.day == 1)]['date']
    plt.figure(figsize=(12, 5))
    axs = sns.lineplot(df, x='date', y='total_demand', lw=1, errorbar=None, color=vc['all'], zorder=2, label='Power Demand')
    sns.lineplot(peak_df, x='date', y='total_demand', lw=1, errorbar=None, color=vc['peak'], zorder=3, ax=axs, label='Peak Demand')
    for l in jan_1:
        plt.axvline(x=l, color='lightgrey', linestyle='--', lw=0.5, zorder=1)
    plt.title('Demand')
    axs.legend()
    plt.savefig(img_folder / 'demand_time')
    plt.close()


def avg_stats_across_years(d_frame, var, axs, ylab, xlab=False):
    """
    calculate aggregated statistics for each day across all years for the specified variable
    plot statistics on specified axis
    """
    stats_colours = {'mean': vc['var2'], 'min': vc['var3'], 'max': vc['var5']}
    daily_stats = d_frame.groupby(['month', 'day'])[var].agg(['mean', 'min', 'max']).reset_index()
    daily_stats['date'] = pd.to_datetime('2012' + '-' + daily_stats['month'].astype(str) + '-' + daily_stats['day'].astype(str), format='%Y-%m-%d')
    daily_stats = pd.melt(daily_stats, id_vars=['date'], value_vars=['mean', 'min', 'max'], var_name='statistic')
    sns.lineplot(daily_stats, x='date', y='value', hue='statistic', ax=axs, palette=stats_colours)
    axs.set_xlim(daily_stats['date'].min(), daily_stats['date'].max()) 
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs.set_ylabel(ylab)
    axs.grid(True, axis='x', linestyle='--', alpha=0.5)
    if not xlab:
        axs.set_xlabel('')


def get_month_day(df): 
    """
    extract month and day from date column
    """
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df


def daily_stats(df, peak_df, img_folder):
    """
    plot variable statistics averaged between years for all numerical variables
    compare all demand and peak demand statistics
    """
    df = get_month_day(df)
    peak_df = get_month_day(peak_df)
    col_list = df.columns.tolist()[1:6]
    for col in col_list:
        fig, ax = plt.subplots(2, 1, figsize=(8,6))
        avg_stats_across_years(df, col, ax[0], 'Electricity Demand')
        avg_stats_across_years(peak_df, col, ax[1], 'Peak Electricity Demand', True)

        handles, labels = ax[0].get_legend_handles_labels()
        for axs in ax.flat:
            axs.legend_.remove()
        fig.legend(handles, labels, loc='upper right')
        ax[0].set_title(total_labels[col])
        img_name = col + '_statistics'
        plt.suptitle('Average Variable Statistics')
        plt.tight_layout()
        img_name = col + '_statistics'
        plt.savefig(img_folder / img_name)
        plt.close()


def aemo_forecast(forecast, img_folder):
    """
    residual plots for AEMO forecast
    """
    fig, axs = plt.subplots(1, 2, figsize=(12,6), width_ratios=[2, 1])
    forecast['datetime'] = pd.to_datetime(forecast['datetime'])
    forecast['forecast_datetime'] = pd.to_datetime(forecast['forecast_datetime'])
    forecast['resid'] = forecast['total_demand'] - forecast['forecast_demand']
    sns.lineplot(forecast, x='datetime', y='resid', ax=axs[0], linewidth=0.5, color=vc['var2'])
    sns.scatterplot(forecast, x='forecast_demand', y='resid', ax=axs[1], alpha=0.2, color=vc['var2'])
    axs[1].set_ylabel('')
    plt.suptitle('AEMO Forecast Residuals')
    for ax in axs:
        ax.grid(color='grey', linestyle='--', linewidth=0.25, alpha=0.4)
        ax.axhline(y=0, color='grey', linestyle=':', alpha=0.7, zorder=1)
    plt.tight_layout()
    plt.savefig(img_folder / 'aemo_forecast_residuals')
    plt.close()


def statistic_reports(df, peak_df, col):
    print('All Demand:')
    print(df[col].describe())
    print(df[col].skew())
    print(df[col].kurt())
    print('\nPeak Demand:')
    print(peak_df[col].describe())
    print(peak_df[col].skew())
    print(peak_df[col].kurt())




cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'
img_folder = root_folder / 'figures'

demand_df = pd.read_csv(data_folder / 'all_data_30min.csv')
demand_df['datetime'] = pd.to_datetime(demand_df['datetime'])
demand_df['time'] = demand_df['datetime'].dt.time
demand_df['time'] = pd.to_datetime(demand_df['time'], format='%H:%M:%S')
demand_df['date'] = demand_df['datetime'].dt.date
demand_df['date'] = pd.to_datetime(demand_df['date'], format='%Y-%m-%d')
demand_df = demand_df[demand_df['date']!='2010-01-01']
seasons_dict={1: 'Summer', 2: 'Summer', 3: 'Autumn', 4: 'Autumn', 5: 'Autumn', 6: 'Winter',
                  7: 'Winter', 8: 'Winter', 9: 'Spring', 10: 'Spring', 11: 'Spring', 12: 'Summer'}
demand_df['season'] = demand_df['date'].dt.month.map(seasons_dict)

# get corresponding data to daily peak demand
max_idx = demand_df.groupby('date')['total_demand'].idxmax()
peak_demand_df = demand_df.loc[max_idx].reset_index(drop=True)

forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')


# explore relationships between variables
#scatterplot_matrix(demand_df, img_folder, 'demand_var_comparison', vc['all'])
#scatterplot_matrix(peak_demand_df, img_folder, 'peak_demand_var_comparison', vc['peak'])
#corr_matrix (demand_df, peak_demand_df, img_folder)

# explore variable distributions
var_distributions(demand_df, peak_demand_df, img_folder)

# explore temporal distributions
#demand_time(peak_demand_df, img_folder)
#demand_time_all(demand_df, peak_demand_df, img_folder)
#daily_stats(demand_df, peak_demand_df, img_folder)

# plot AEMO forecast distributions
#aemo_forecast(forecast_demand_df, img_folder)



#statistic_reports(demand_df, peak_demand_df, 'pv_capacity')

