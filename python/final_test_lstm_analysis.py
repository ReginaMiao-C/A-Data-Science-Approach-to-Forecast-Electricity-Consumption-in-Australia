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
    df['true_peak_time'] = pd.to_datetime(df['true_peak_time']).dt.round('30min')
    df['pred_peak_time'] = pd.to_datetime(df['pred_peak_time']).dt.round('30min')

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

def plot_results(df, res_path, model_name, label='LSTM Prediction'):
    df['resid'] = df['true_peak'] - df['pred_peak']
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(df, x='date', y='true_peak', color='orange', ax = axs[0], label='True Value')
    sns.lineplot(df, x='date', y='pred_peak', color='sienna', ax = axs[0], linestyle='--', label=label, linewidth=1)
    axs[0].legend()
    sns.scatterplot(df, x='date', y='resid', color='sienna', ax = axs[1], alpha=0.6)
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
    plt.suptitle('Peak Demand Forecast')
    plt.tight_layout()
    img_name = 'pred_vs_actual_' + model_name
    plt.savefig(res_path / img_name)
    plt.close()

    df['time_resid'] = (df['true_peak_time'] - df['pred_peak_time']).dt.total_seconds() / 60
    df['true_peak_time'] = pd.to_datetime('1970-01-01 ' + df['true_peak_time'].dt.time.astype(str))
    df['pred_peak_time'] = pd.to_datetime('1970-01-01 ' + df['pred_peak_time'].dt.time.astype(str))
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(df, x='date', y='true_peak_time', color='orange', label='True Value', ax = axs[0])
    sns.lineplot(df, x='date', y='pred_peak_time', color='sienna',  ax = axs[0], linestyle='--', label=label, linewidth=1)
    axs[0].legend()
    sns.scatterplot(df, x='date', y='time_resid', color='sienna', ax = axs[1], alpha=0.6, zorder=2)
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
    plt.suptitle('Time of Peak Demand Forecast')
    plt.tight_layout()
    img_name = 'pred_vs_actual_time_' + model_name
    plt.savefig(res_path / img_name)
    plt.close()


def plot_resids(df, res_path, model_name, model):
    df['pred_resid'] = df['true_peak'] -  df['pred_peak'] 
    df['aemo_resid'] = df['true_peak'] -  df['aemo_pred_peak'] 
    df = df[['date', 'pred_resid', 'aemo_resid']]
    df_diff = df.copy()
    df_diff['differences'] = df_diff['aemo_resid'].abs() - df_diff['pred_resid'].abs()

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sns.scatterplot(df_diff, x='date', y='differences', color='salmon', ax=axs[1], alpha=0.7)

    df = df.rename(columns={'pred_resid': model, 'aemo_resid': 'AEMO'})
    df = df.melt(id_vars='date', value_vars=[model, 'AEMO'], var_name='Prediction', value_name='Peak Electricity Demand')
    pred_colors = {model:'peru', 'AEMO':'brown'}
    sns.lineplot(df, x='date', y='Peak Electricity Demand', hue='Prediction', palette=pred_colors, ax=axs[0])
    axs[0].axhline(0, color='grey', linestyle=':', zorder=1) 
    axs[1].axhline(0, color='grey', linestyle=':', zorder=1) 
    axs[0].set_ylabel('Residuals (MW)')
    label_str ='Absolute Residual Differences (AEMO - ' + model + ')'
    axs[1].set_ylabel(label_str)
    axs[0].set_xlabel('')
    axs[1].set_xlabel('Date')
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[0].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    axs[0].grid(axis='x', linestyle='--', color='lightgrey', zorder=0, which='both')
    axs[1].grid(axis='x', linestyle='--', color='lightgrey', zorder=0)
    title_str ='Forecast Residuals (AEMO vs ' + model + ')'
    plt.suptitle(title_str)
    plt.tight_layout()
    img_name = 'residual_comparison_' + model_name
    plt.savefig(res_path / img_name)
    plt.close()
    print(df_diff['differences'].sum())


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')

# Ensemble/Ensemble_nr/LSTM/SARIMAX/AEMO
model = 'Ensemble_nr'

if model == 'LSTM':
    # best final model
    csv_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Var Dropout'
    save_path = csv_path

    df = pd.read_csv(csv_path / 'dropped_8_test.csv')
    forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')
    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, '8_test')
    plot_resids(resid_df, save_path, '8_test', model)


elif model == 'Ensemble':
    # LSTM + SARIMAX ensemble
    csv_path = data_folder
    save_path = root_folder / 'Results' / 'ensemble'
    df = pd.read_csv(csv_path / 'test_ensemble_final.csv')
    df = df.drop(columns=['sarimax_peak_value_predicted', 'sarimax_peak_time_predicted','lstm_peak_value_predicted', 'lstm_peak_time_predicted',])
    df = df.rename(columns={'actual_peak_value': 'true_peak', 'actual_peak_time': 'true_peak_time', 'ensemble_peak_value_predicted': 'pred_peak', 'ensemble_peak_time_predicted': 'pred_peak_time'})

    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, 'ensemble_test')
    plot_resids(resid_df, save_path, 'ensemble_test', model)


elif model == 'Ensemble_nr':
    # ensemble combined with AEMO predictions
    csv_path = cwd / 'AEMO_files'
    save_path = root_folder / 'Results' / 'ensemble'
    df = pd.read_csv(csv_path / 'test_ensemble_final_norestriciton.csv')
    df = df.drop(columns=['sarimax_peak_value_predicted', 'sarimax_peak_time_predicted','lstm_peak_value_predicted', 'lstm_peak_time_predicted',])
    df = df.rename(columns={'actual_peak_value': 'true_peak', 'actual_peak_time': 'true_peak_time', 'ensemble_peak_value_predicted': 'pred_peak', 'ensemble_peak_time_predicted': 'pred_peak_time'})

    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, 'ensemble_test_nr')
    plot_resids(resid_df, save_path, 'ensemble_test_nr', model)


elif model == 'SARIMAX':
    csv_path = cwd / 'SARIMAX' / "data"
    save_path = root_folder / 'Results' / 'SARIMAX'
    df = pd.read_csv(csv_path / 'daily_data_with_exog2026-04-24.csv')
    df = df[['eval_date', 'peak_actual_afternoon', 'peak_predicted_afternoon_mean', 'time_of_peak_actual_afternoon', 'time_of_peak_predicted_afternoon', 'peak_actual_morning', 'peak_predicted_morning_mean', 'time_of_peak_actual_morning', 'time_of_peak_predicted_morning']]
    df['eval_date'] = pd.to_datetime(df['eval_date'])
    #df = df[df['eval_date'].dt.year == 2020]
    df = df[df['eval_date'] >= '03/12/2020']

    df['time_of_peak_actual_afternoon'] = df['time_of_peak_actual_afternoon'].str[11:] + ' PM'
    df['time_of_peak_actual_afternoon'] = df['time_of_peak_actual_afternoon'].replace('00:00:00 PM', '12:00:00 PM')
    df['time_of_peak_actual_afternoon'] = pd.to_datetime(df['time_of_peak_actual_afternoon'], format='%I:%M:%S %p').dt.strftime('%H:%M')
    df['time_of_peak_predicted_afternoon'] = df['time_of_peak_predicted_afternoon'].str[11:] + ' PM'
    df['time_of_peak_predicted_afternoon'] = pd.to_datetime(df['time_of_peak_predicted_afternoon'], format='%I:%M:%S %p').dt.strftime('%H:%M')

    df['time_of_peak_actual_morning'] = df['time_of_peak_actual_morning'].str[11:] + ' AM'
    df['time_of_peak_actual_morning'] = df['time_of_peak_actual_morning'].replace('00:00:00 AM', '12:00:00 AM')
    df['time_of_peak_actual_morning'] = pd.to_datetime(df['time_of_peak_actual_morning'], format='%I:%M:%S %p').dt.strftime('%H:%M')
    df['time_of_peak_predicted_morning'] = df['time_of_peak_predicted_morning'].str[11:] + ' AM'
    df['time_of_peak_predicted_morning'] = df['time_of_peak_predicted_morning'].replace('00:00:00 AM', '12:00:00 AM')
    df['time_of_peak_predicted_morning'] = pd.to_datetime(df['time_of_peak_predicted_morning'], format='%I:%M:%S %p').dt.strftime('%H:%M')


    peak_comp = df['peak_actual_afternoon'] > df['peak_actual_morning']

    df_final = pd.DataFrame({'date': df['eval_date'], 'true_peak': np.where(peak_comp, df['peak_actual_afternoon'], df['peak_actual_morning']), 
                             'true_peak_time': np.where(peak_comp, df['time_of_peak_actual_afternoon'], df['time_of_peak_actual_morning']), 
                             'pred_peak': np.where(peak_comp, df['peak_predicted_afternoon_mean'], df['peak_predicted_morning_mean']), 
                             'pred_peak_time': np.where(peak_comp, df['time_of_peak_predicted_afternoon'], df['time_of_peak_predicted_morning'])})

    forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')
    df_final, resid_df = process_dfs(df_final, forecast_demand_df)
    plot_results(df_final, save_path, 'sarimax_test', "SARIMAX Predictions")
    plot_resids(resid_df, save_path, 'sarimax_test', model)

elif model == 'AEMO':
    csv_path = root_folder / 'data'
    save_path = root_folder / 'Results'
    df = pd.read_csv(csv_path / 'aemo_peak_and_time.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '03/12/2020']
    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, 'aemo')