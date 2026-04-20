import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
import public_holidays as ph
import matplotlib.pyplot as plt
import seaborn as sns
from colour_dict import demand_cols as vc

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

res_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Test'


df_all = pd.read_csv(res_path / 'all_final.csv')
df_all['date'] = pd.to_datetime(df_all['date'], format='%Y-%m-%d')
df_all['true_peak_time'] = pd.to_datetime(df_all['true_peak_time'], format='%H:%M:%S')
df_all['pred_peak_time'] = pd.to_datetime(df_all['pred_peak_time'], format='%H:%M:%S')


def plot_res(df, title):
    df['resid'] = df['true_peak'] - df['pred_peak']
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(df, x='date', y='true_peak', color='orange', ax = axs[0])
    sns.lineplot(df, x='date', y='pred_peak', color='brown', ax = axs[0], linestyle='--')
    sns.scatterplot(df, x='date', y='resid', color='grey', ax = axs[1])
    #plt.suptitle(title)
    plt.savefig(res_path / 'pred_vs_actual_final')
    plt.close()
    sns.lineplot(df, x='date', y='true_peak_time', color='orange')
    sns.lineplot(df, x='date', y='pred_peak_time', color='brown')
    plt.savefig(res_path / 'pred_residuals_final')
    plt.close()

plot_res(df_all, 'All 2020')

forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')

forecast_demand_df['datetime'] = pd.to_datetime(forecast_demand_df['datetime'])
forecast_demand_df['time'] = forecast_demand_df['datetime'].dt.time
forecast_demand_df['time'] = pd.to_datetime(forecast_demand_df['time'], format='%H:%M:%S')
forecast_demand_df['date'] = forecast_demand_df['datetime'].dt.date
forecast_demand_df['date'] = pd.to_datetime(forecast_demand_df['date'], format='%Y-%m-%d')



forecast_demand_df_filtered = forecast_demand_df[['date', 'time', 'forecast_demand']]
forecast_demand_df_filtered = forecast_demand_df_filtered.rename(columns={'time': 'aemo_peak_time', 'forecast_demand': 'aemo_pred_peak'})
df_all_filtered = df_all[['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time']]

df_comb = pd.merge(df_all_filtered, forecast_demand_df_filtered, on='date', how='left')
df_comb['pred_resid'] = df_comb['pred_peak'] - df_comb['true_peak']
df_comb['aemo_resid'] = df_comb['aemo_pred_peak'] - df_comb['true_peak']

df_resid = df_comb[['date', 'pred_resid', 'aemo_resid']]
df_resid_diff = df_resid.copy()
df_resid_diff['differences'] = df_resid_diff['aemo_resid'] - df_resid_diff['pred_resid']

sns.lineplot(df_resid_diff, x='date', y='differences', color='sienna')
plt.axhline(0, color='lightgrey', linestyle='--', zorder=1) 
plt.savefig(res_path / 'residual_differences_final')
plt.close()
print(df_resid_diff['differences'].sum())

df_resid = df_resid.rename(columns={'pred_resid': 'LSTM', 'aemo_resid': 'AEMO'})
df_resid = df_resid.melt(id_vars='date', value_vars=['LSTM', 'AEMO'], var_name='Prediction', value_name='Peak Power')

pred_colors = {'LSTM':'brown', 'AEMO':'salmon'}
sns.lineplot(df_resid, x='date', y='Peak Power', hue='Prediction', palette=pred_colors)
plt.axhline(0, color='lightgrey', linestyle='--', zorder=1) 
plt.savefig(res_path / 'residual_comparison_final')
plt.close()
