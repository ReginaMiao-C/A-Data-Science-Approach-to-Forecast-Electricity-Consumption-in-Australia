import pandas as pd
from pathlib import Path
import sys
import numpy as np
import public_holidays as ph
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error


def process_dfs(df, aemo_df, df_format='%Y-%m-%d'):
    """
    returns model and AEMO predictions in consistent formats for plotting results
    """
    # datetime processing
    df['date'] = pd.to_datetime(df['date'], format=df_format)
    df['true_peak_time'] = pd.to_datetime(df['true_peak_time']).dt.round('30min')
    df['pred_peak_time'] = pd.to_datetime(df['pred_peak_time']).dt.round('30min')

    aemo_df['datetime'] = pd.to_datetime(aemo_df['datetime'])
    aemo_df['time'] = aemo_df['datetime'].dt.time
    aemo_df['time'] = pd.to_datetime(aemo_df['time'], format='%H:%M:%S')
    aemo_df['date'] = aemo_df['datetime'].dt.date
    aemo_df['date'] = pd.to_datetime(aemo_df['date'], format='%Y-%m-%d')

    #filter and combine dataframes
    aemo_df_filtered = aemo_df[['date', 'time', 'forecast_demand']].copy()
    aemo_df_filtered = aemo_df_filtered.rename(columns={'time': 'aemo_peak_time', 'forecast_demand': 'aemo_pred_peak'})
    df_filtered = df[['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time']].copy()
    df_comb = pd.merge(df_filtered, aemo_df_filtered, on='date', how='left')
    return df, df_comb

def display_metrics(df):
    time_cols=['true_peak_time', 'pred_peak_time']
    df[time_cols] = df[time_cols].astype(int) #time in nanoseconds
    df[time_cols] = (df[time_cols]/1e9)/60 #time in mins
    print('MAGNITUDE')
    print('MSE: ', mean_squared_error(df['true_peak'], df['pred_peak']))
    print('MAE: ', mean_absolute_error(df['true_peak'], df['pred_peak']))
    print('MAPE: ', (mean_absolute_percentage_error(df['true_peak'], df['pred_peak'])*100))
    print('TIME')
    print('MSE: ', mean_squared_error(df['true_peak_time'], df['pred_peak_time']))
    print('MAE: ', mean_absolute_error(df['true_peak_time'], df['pred_peak_time']))

def plot_results(df, res_path, model_name, model):
    """
    plots true and predicted values over time for daily peak demand magnitude and time
    """
    # calculate magnitude residuals
    df['resid'] = df['true_peak'] - df['pred_peak']
    # plot true and predicted magnitudes
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(df, x='date', y='true_peak', color='orange', ax = axs[0], label='True Value')
    sns.lineplot(df, x='date', y='pred_peak', color='sienna', ax = axs[0], linestyle='--', label=model_name, linewidth=1)
    axs[0].legend()
    # plot residuals
    sns.scatterplot(df, x='date', y='resid', color='sienna', ax = axs[1], alpha=0.6)
    # format plots
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
    img_name = 'pred_vs_actual_' + model
    plt.savefig(res_path / img_name)
    plt.close()

    # calculate time residuals
    
    df['time_resid'] = (df['true_peak_time'] - df['pred_peak_time']).dt.total_seconds() / 60
    # format times
    df['true_peak_time'] = pd.to_datetime('1970-01-01 ' + df['true_peak_time'].dt.time.astype(str))
    df['pred_peak_time'] = pd.to_datetime('1970-01-01 ' + df['pred_peak_time'].dt.time.astype(str))
    # plot true and predicted times
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(df, x='date', y='true_peak_time', color='orange', label='True Value', ax = axs[0])
    sns.lineplot(df, x='date', y='pred_peak_time', color='sienna',  ax = axs[0], linestyle='--', label=model_name, linewidth=1)
    axs[0].legend()
    # plot residuals
    sns.scatterplot(df, x='date', y='time_resid', color='sienna', ax = axs[1], alpha=0.6, zorder=2)
    # format plots
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
    img_name = 'pred_vs_actual_time_' + model
    plt.savefig(res_path / img_name)
    plt.close()


def plot_resids(df, res_path, model_name, model):
    """
    plots differences between residuals over time for AEMO and model magnitude predictions
    """
    # calculate residuals
    df['pred_resid'] = df['true_peak'] -  df['pred_peak'] 
    df['aemo_resid'] = df['true_peak'] -  df['aemo_pred_peak'] 
    # format data
    df = df[['date', 'pred_resid', 'aemo_resid']]
    df_diff = df.copy()
    df_diff['differences'] = df_diff['aemo_resid'].abs() - df_diff['pred_resid'].abs()

    # plot residuals over time
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    sns.scatterplot(df_diff, x='date', y='differences', color='salmon', ax=axs[1], alpha=0.7)
    # plot residual differences over time
    df = df.rename(columns={'pred_resid': model_name, 'aemo_resid': 'AEMO'})
    df = df.melt(id_vars='date', value_vars=[model_name, 'AEMO'], var_name='Prediction', value_name='Peak Electricity Demand')
    pred_colors = {model_name:'peru', 'AEMO':'brown'}
    sns.lineplot(df, x='date', y='Peak Electricity Demand', hue='Prediction', palette=pred_colors, ax=axs[0])
    # format plots
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
    title_str ='Forecast Residuals (AEMO vs ' + model_name + ')'
    plt.suptitle(title_str)
    plt.tight_layout()
    img_name = 'residual_comparison_' + model
    plt.savefig(res_path / img_name)
    plt.close()
    print('Sum of absolute residual difference', df_diff['differences'].sum())


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')

# select model:
# Ensemble/Ensemble_nr/LSTM/SARIMAX/AEMO
model = 'AEMO'

if model == 'LSTM':
    csv_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Var Dropout'
    save_path = root_folder / 'Results' / 'LSTM'

    df = pd.read_csv(csv_path / 'dropped_8_test.csv')
    forecast_demand_df = pd.read_csv(data_folder / 'peak_forecasts.csv')
    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, 'LSTM', model)
    plot_resids(resid_df, save_path, 'LSTM', model)
    display_metrics(df)


elif model == 'Ensemble':
    # LSTM-SARIMAX ensemble
    csv_path = cwd / 'Ensemble_files'
    save_path = root_folder / 'Results' / 'ensemble'
    df = pd.read_csv(csv_path / 'test_ensemble_final_without_AEMO.csv')
    df = df.drop(columns=['sarimax_peak_value_predicted', 'sarimax_peak_time_predicted','lstm_peak_value_predicted', 'lstm_peak_time_predicted',])
    df = df.rename(columns={'actual_peak_value': 'true_peak', 'actual_peak_time': 'true_peak_time', 'ensemble_peak_value_predicted': 'pred_peak', 'ensemble_peak_time_predicted': 'pred_peak_time'})

    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, 'Ensemble', model)
    plot_resids(resid_df, save_path, 'Ensemble', model)
    display_metrics(df)



elif model == 'Ensemble_nr':
    # LSTM-SARIMAX-AEMO ensemble
    csv_path = cwd / 'Ensemble_files'
    save_path = root_folder / 'Results' / 'ensemble'
    df = pd.read_csv(csv_path / 'test_ensemble_final_with_AEMO.csv')
    df = df.drop(columns=['sarimax_peak_value_predicted', 'sarimax_peak_time_predicted','lstm_peak_value_predicted', 'lstm_peak_time_predicted',])
    df = df.rename(columns={'actual_peak_value': 'true_peak', 'actual_peak_time': 'true_peak_time', 'ensemble_peak_value_predicted': 'pred_peak', 'ensemble_peak_time_predicted': 'pred_peak_time'})

    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, 'Ensemble', model)
    plot_resids(resid_df, save_path, 'Ensemble', model)
    display_metrics(df)


elif model == 'SARIMAX':
    csv_path = cwd / 'SARIMAX' / "data"
    save_path = root_folder / 'Results' / 'SARIMAX'
    df = pd.read_csv(csv_path / 'daily_data_with_exog2026-04-24.csv')

    # format timeseries data 
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
    plot_results(df_final, save_path, 'sarimax_test', model)
    plot_resids(resid_df, save_path, 'sarimax_test', model)
    display_metrics(df_final)

elif model == 'AEMO':
    # can be used for plotting peak magnitudes and times over time
    csv_path = root_folder / 'data'
    save_path = root_folder / 'Results'
    df = pd.read_csv(csv_path / 'aemo_peak_and_time.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '03/12/2020']
    df, resid_df = process_dfs(df, forecast_demand_df)
    plot_results(df, save_path, 'aemo', model)
    display_metrics(df)

"""
# calculate test stats
csv_path = cwd / 'Ensemble_files'
df = pd.read_csv(csv_path / 'test_ensemble_final_with_AEMO.csv')

print(df.head())
print(df.columns)
time_cols = ['sarimax_peak_time_predicted', 'lstm_peak_time_predicted',
       'actual_peak_time', 'AEMO_peak_time_predicted', 'ensemble_peak_time_predicted', ]
df[time_cols] = df[time_cols].apply(pd.to_datetime)
df[time_cols] = df[time_cols].astype(int) #time in nanoseconds
df[time_cols] = (df[time_cols]/1e9)/60 #time in mins
df['date'] = pd.to_datetime(df['date'])

print('\nSARIMAX')
print('MAGNITUDE')
print('MSE: ', mean_squared_error(df['actual_peak_value'], df['sarimax_peak_value_predicted']))
print('MAE: ', mean_absolute_error(df['actual_peak_value'], df['sarimax_peak_value_predicted']))
print('MAPE: ', (mean_absolute_percentage_error(df['actual_peak_value'], df['sarimax_peak_value_predicted'])*100))
print('TIME')
print('MSE: ', mean_squared_error(df['actual_peak_time'], df['sarimax_peak_time_predicted']))
print('MAE: ', mean_absolute_error(df['actual_peak_time'], df['sarimax_peak_time_predicted']))

print('\n\nLSTM')
print('MAGNITUDE')
print('MSE: ', mean_squared_error(df['actual_peak_value'], df['lstm_peak_value_predicted']))
print('MAE: ', mean_absolute_error(df['actual_peak_value'], df['lstm_peak_value_predicted']))
print('MAPE: ', (mean_absolute_percentage_error(df['actual_peak_value'], df['lstm_peak_value_predicted'])*100))
print('TIME')
print('MSE: ', mean_squared_error(df['actual_peak_time'], df['lstm_peak_time_predicted']))
print('MAE: ', mean_absolute_error(df['actual_peak_time'], df['lstm_peak_time_predicted']))


df2 = pd.read_csv(csv_path / 'lstm_test.csv')
print('\nMSE: ', mean_squared_error(df2['true_demand'], df2['lstm_pred_demand']))
print('MAE: ', mean_absolute_error(df2['true_demand'], df2['lstm_pred_demand']))
print('MAPE: ', (mean_absolute_percentage_error(df2['true_demand'], df2['lstm_pred_demand'])*100))

csv_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Var Dropout'

df3 = pd.read_csv(csv_path / 'dropped_8_test.csv')

print('\nMSE: ', mean_squared_error(df3['true_peak'], df3['pred_peak']))
print('MAE: ', mean_absolute_error(df3['true_peak'], df3['pred_peak']))
print('MAPE: ', (mean_absolute_percentage_error(df3['true_peak'], df3['pred_peak'])*100))"""