import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'
img_folder = root_folder / 'figures'

df = pd.read_csv(data_folder / 'all_data_30min.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df['time'] = df['datetime'].dt.time
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['date'] = df['datetime'].dt.date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df[df['date']!='2010-01-01']

# get corresponding data to daily peak demand
max_idx = df.groupby('date')['total_demand'].idxmax()
peak_df = df.loc[max_idx].reset_index(drop=True)


# TODO: get histogram of peak times






## RELATIONSHIPS BETWEEN VARIABLES ##
"""
# scatterplot grid of all variables
p = sns.pairplot(df, plot_kws={'alpha': 0.02, 's': 10})
plt.subplots_adjust(bottom=0.1)
plt.savefig(img_folder / 'demand_var_comparison')
plt.close()

p = sns.pairplot(peak_df, plot_kws={'alpha': 0.2, 's': 10})
plt.subplots_adjust(bottom=0.1)
plt.savefig(img_folder / 'peak_demand_var_comparison')
plt.close()
# NOTE: extreme values of max and min temperatures (to a lesser extent solar exposure) correlate with high peak demand
"""

"""
=
cm = df.corr(numeric_only=True)
peak_cm = peak_df.corr(numeric_only=True)
fig, axs = plt.subplots(1, 2, figsize=(11,6))
sns.heatmap(cm, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axs[0], cbar=False)
sns.heatmap(peak_cm, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axs[1], cbar=False)
axs[0].set_title('All Demand')
axs[1].set_title('Peak Demand')
axs[1].tick_params(axis='y', labelleft=False)
axs[0].tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.suptitle('Correlation Between Variables')
plt.savefig(img_folder / 'corr_matrices')
plt.close()
"""
"""
## VARIABLE DISTRIBUTION ##
fig, axs = plt.subplots(2, 5, figsize=(12,6))
col_list = df.columns.tolist()[1:6]
for col in col_list:
    col_idx = col_list.index(col)
    sns.violinplot(df[col], ax=axs[0][col_idx], color='cadetblue')
    sns.violinplot(peak_df[col], ax=axs[1][col_idx], color='palevioletred')
    ymin, ymax = axs[0][col_idx].get_ylim()
    axs[1][col_idx].set_ylim(ymin, ymax)
    axs[0][col_idx].set_title(col)
    axs[0][col_idx].set_ylabel('')
    axs[1][col_idx].set_ylabel('')
fig.text(0.01, 0.68, 'All Demand', va='center', rotation=90, fontsize=12)
fig.text(0.01, 0.23, 'Peak Demand', va='center', rotation=90, fontsize=12)
plt.suptitle('Variable Distributions')
plt.tight_layout()
fig.subplots_adjust(left=0.06) 
plt.savefig(img_folder / 'var_violinplots')
plt.close()

"""


"""
fig, axs = plt.subplots(2, 5, figsize=(12,6))
col_list = df.columns.tolist()[1:6]
for col in col_list:
    col_idx = col_list.index(col)
    sns.histplot(df, x=col, ax=axs[0][col_idx], color='cadetblue')
    sns.histplot(peak_df, x=col, ax=axs[1][col_idx], color='palevioletred')
    axs[0][col_idx].set_title(col)
    axs[0][col_idx].set_ylabel('')
    axs[1][col_idx].set_ylabel('')
fig.text(0.01, 0.68, 'All Demand', va='center', rotation=90, fontsize=12)
fig.text(0.01, 0.23, 'Peak Demand', va='center', rotation=90, fontsize=12)
plt.suptitle('Variable Counts')
plt.tight_layout()
fig.subplots_adjust(left=0.1) 
plt.show()
plt.close()

"""
col_list = df.columns.tolist()[1:6]
for col in col_list:
    fig, axs = plt.subplots(1, 2, figsize=(8,5))
    sns.histplot(df, x=col, ax=axs[0], color='cadetblue', bins=50)
    sns.histplot(peak_df, x=col, ax=axs[1], color='palevioletred', bins=50)
    xmin, xmax = axs[0].get_xlim()
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylabel('')
    axs[0].set_title('All Demand')
    axs[1].set_title('Peak Demand')
    img_name = col + '_histograms'
    plt.savefig(img_folder / img_name)


sys.exit()

df['demand'] = 'all'
peak_df['demand'] = 'peak'
comb_df = pd.concat([df, peak_df], ignore_index=True)


# peak demand vs time
jan_1 = df[(df['date'].dt.month == 1) & (df['date'].dt.day == 1)]['date']

sns.lineplot(df, x='date', y='peak_power')
for l in jan_1:
    plt.axvline(x=l, color='darkgrey', linestyle='--', lw=0.5, zorder=1)
plt.show()
plt.clf()
# NOTE: extreme demand values showwn in summer, lowest min values in winter



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
plt.clf()

cm = df.corr(numeric_only=True)
print(cm)
sns.heatmap(cm, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()  
plt.clf()