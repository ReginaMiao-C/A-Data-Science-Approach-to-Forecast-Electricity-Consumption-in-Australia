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
import matplotlib.dates as mdates



def process_dfs(df, aemo_df, df_format='%Y-%m-%d'):
    df['date'] = pd.to_datetime(df['date'], format=df_format)
    df['true_peak_time'] = pd.to_datetime(df['true_peak_time'])
    df['pred_peak_time'] = pd.to_datetime(df['pred_peak_time'])

    aemo_df['datetime'] = pd.to_datetime(aemo_df['datetime'])
    aemo_df['time'] = aemo_df['datetime'].dt.time
    aemo_df['time'] = pd.to_datetime(aemo_df['time'], format='%H:%M:%S')
    aemo_df['date'] = aemo_df['datetime'].dt.date
    aemo_df['date'] = pd.to_datetime(aemo_df['date'], format='%Y-%m-%d')

    aemo_df_filtered = aemo_df[['date', 'time', 'forecast_demand']].copy()
    aemo_df_filtered = aemo_df_filtered.rename(columns={'time': 'aemo_peak_time', 'forecast_demand': 'aemo_pred_peak'})
    df_filtered = df[['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time']].copy()
    df_comb = pd.merge(df_filtered, aemo_df_filtered, on='date', how='left')
    return df, df_comb

def plot_results(df, res_path, model_name):
    df['resid'] = df['true_peak'] - df['pred_peak']
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(df, x='date', y='true_peak', color='orange', ax = axs[0], label='True Value')
    sns.lineplot(df, x='date', y='pred_peak', color='brown', ax = axs[0], linestyle='--', label='LSTM Prediction', linewidth=1)
    axs[0].legend()
    sns.scatterplot(df, x='date', y='resid', color='brown', ax = axs[1], alpha=0.6)
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[0].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    axs[1].axhline(0, color='grey', linestyle=':', zorder=1) 
    axs[0].grid(axis='x', linestyle='--', color='lightgrey', zorder=1, which='both')
    axs[1].grid(axis='x', linestyle='--', color='lightgrey', zorder=1, which='both')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('Date')
    axs[0].set_ylabel('Peak Electricity Demand (MW)')
    axs[1].set_ylabel('Residuals (MW)')
    img_name = 'pred_vs_actual_' + model_name
    plt.savefig(res_path / img_name)
    plt.close()

    df['time_resid'] = (df['true_peak_time'] - df['pred_peak_time']).dt.total_seconds() / 60
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(df, x='date', y='true_peak_time', color='orange', label='True Value', ax = axs[0])
    sns.lineplot(df, x='date', y='pred_peak_time', color='brown',  ax = axs[0], linestyle='--', label='LSTM Prediction', linewidth=1)
    axs[0].legend()
    sns.scatterplot(df, x='date', y='time_resid', color='brown', ax = axs[1], alpha=0.6, zorder=2)
    axs[1].axhline(0, color='grey', linestyle=':', zorder=1) 
    axs[0].yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axs[0].yaxis.set_major_locator(mdates.HourLocator(interval=4))
    axs[0].yaxis.set_minor_locator(mdates.HourLocator(interval=2))
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[0].set_xlabel('')
    axs[1].set_xlabel('Date')
    axs[0].set_ylabel('Peak Time')
    axs[1].set_ylabel('Residuals (Minutes)')
    axs[0].grid(axis='y', linestyle='--', color='lightgrey', zorder=1, which='both')
    axs[1].grid(axis='y', linestyle='--', color='lightgrey', zorder=0)
    img_name = 'pred_vs_actual_time_' + model_name
    plt.savefig(res_path / img_name)
    plt.close()


def plot_resids(df, res_path, model_name):
    df['pred_resid'] = df['true_peak'] -  df['pred_peak'] 
    df['aemo_resid'] = df['true_peak'] -  df['aemo_pred_peak'] 
    df = df[['date', 'pred_resid', 'aemo_resid']]
    df_diff = df.copy()
    df_diff['differences'] = df_diff['aemo_resid'].abs() - df_diff['pred_resid'].abs()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sns.scatterplot(df_diff, x='date', y='differences', color='sienna', ax=axs[1], alpha=0.7)

    df = df.rename(columns={'pred_resid': 'LSTM', 'aemo_resid': 'AEMO'})
    df = df.melt(id_vars='date', value_vars=['LSTM', 'AEMO'], var_name='Prediction', value_name='Peak Electricity Demand')
    pred_colors = {'LSTM':'brown', 'AEMO':'salmon'}
    sns.lineplot(df, x='date', y='Peak Electricity Demand', hue='Prediction', palette=pred_colors, ax=axs[0])
    axs[0].axhline(0, color='grey', linestyle=':', zorder=1) 
    axs[1].axhline(0, color='grey', linestyle=':', zorder=1) 
    axs[0].set_ylabel('Residuals (MW)')
    axs[1].set_ylabel('Absolute Residual Differences (AEMO - LSTM)')
    axs[0].set_xlabel('')
    axs[0].set_xlabel('Date')
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[0].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    axs[0].grid(axis='x', linestyle='--', color='lightgrey', zorder=0, which='both')
    axs[1].grid(axis='x', linestyle='--', color='lightgrey', zorder=0)

    img_name = 'residual_comparison_' + model_name
    plt.savefig(res_path / img_name)
    plt.close()
    print(df_diff['differences'].sum())


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

csv_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Test'
save_path = csv_path


# best final model
csv_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Var Dropout'
save_path = csv_path

df = pd.read_csv(csv_path / 'dropped_8_test.csv')
forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')

df, resid_df = process_dfs(df, forecast_demand_df)
plot_results(df, save_path, '8_test')
plot_resids(resid_df, save_path, '8_test')
