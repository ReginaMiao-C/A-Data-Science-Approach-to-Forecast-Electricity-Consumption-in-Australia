import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data.csv')
df = df.rename(columns={'Unnamed: 0': 'date'})
df['date'] = pd.to_datetime(df['date'])

"""
sns.pairplot(df, plot_kws={'alpha': 0.2, 's': 10})
plt.show()
# NOTE: extreme values of max and min temperatures (to a lesser extent solar exposure) correlate with high peak demand


## peak demand vs time
jan_1 = df[(df['date'].dt.month == 1) & (df['date'].dt.day == 1)]['date']

sns.lineplot(df, x='date', y='peak_power')
for l in jan_1:
    plt.axvline(x=l, color='darkgrey', linestyle='--', lw=0.5, zorder=1)
plt.show()
# NOTE: extreme demand values showwn in summer, lowest min values in winter
"""

## daily statistics
# TODO: add all vars once PV fixed - this is v much a proof of concept
def avg_stats_across_years(var, axs):
    """
    calculate aggregated statistics for each day across all years for the specified variable
    plot statistics on specified axis
    """
    daily_stats = df.groupby(['month', 'day'])[var].agg(['mean', 'min', 'max']).reset_index()
    daily_stats['date'] = pd.to_datetime('2012' + '-' +daily_stats['month'].astype(str) + '-' + daily_stats['day'].astype(str), format='%Y-%m-%d')
    daily_stats = pd.melt(daily_stats, id_vars=['date'], value_vars=['mean', 'min', 'max'], var_name='statistic')
    sns.lineplot(daily_stats, x='date', y='value', hue='statistic', ax=axs)
    axs.set_xlim(daily_stats['date'].min(), daily_stats['date'].max()) 
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs.set_title(var)

df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

fig,ax = plt.subplots(2, 1, figsize=(10,6))
avg_stats_across_years('peak_power', ax[0])
avg_stats_across_years('min_temperature', ax[1])
plt.show()

