import pandas as pd
from pathlib import Path
import sys

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'
# set source folder for data
source_folder = Path(r'C:\Users\molly\OneDrive\Documents\UNSW\Project\Data')

peak_only = False

# read forecast demand data
if (source_folder / 'forecastdemand_nsw.csv').exists():
    forecast = pd.read_csv(source_folder / 'forecastdemand_nsw.csv')
else:
    if source_folder.exists():
        print('forecastdemand_nsw.csv not found in folder. Exiting...')
        sys.exit()
    else:
        print('Folder not found. Exiting...')
        sys.exit()
# forecast demand preprocessing
forecast = forecast.drop(columns=['PREDISPATCHSEQNO', 'REGIONID', 'PERIODID'])
forecast = forecast.rename(columns={'FORECASTDEMAND': 'forecast_demand', 'LASTCHANGED': 'forecast_datetime', 'DATETIME': 'datetime'})
forecast['datetime'] = pd.to_datetime(forecast['datetime'])

# read demand data
df = pd.read_csv(data_folder / 'all_data_30min.csv')
# demand data preprocessing
df['datetime'] = pd.to_datetime(df['datetime'])
df['time'] = df['datetime'].dt.time
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['date'] = df['datetime'].dt.date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df[df['date']!='2010-01-01']

if peak_only:
    # get corresponding data to daily peak demand
    max_idx = df.groupby('date')['total_demand'].idxmax()
    df = df.loc[max_idx].reset_index(drop=True)

# one to many merge on actual  demand
forecast = pd.merge(df, forecast, on='datetime', how='left')
forecast['forecast_datetime'] = pd.to_datetime(forecast['forecast_datetime'])
forecast['forecast_date'] = forecast['forecast_datetime'].dt.date
forecast['forecast_date'] = pd.to_datetime(forecast['forecast_date'], format='%Y-%m-%d')


# get last forecast from previous day
forecast['prev_date'] = forecast['date'] - pd.Timedelta(days=1)
forecast = forecast[forecast['prev_date'] == forecast['forecast_date']]
forecast = forecast.groupby('datetime').tail(1).reset_index(drop=True)
forecast = forecast.drop(columns=['prev_date', 'rainfall', 'pv_capacity', 'temperature', 'solar_power','time', 'date', 'forecast_date'])

if peak_only:
    forecast.to_csv(data_folder / 'peak_forecasts.csv', index=False)
else:
    forecast.to_csv(data_folder / 'all_forecasts.csv', index=False)

    
    forecast['datetime'] = pd.to_datetime(forecast['datetime'])
    forecast['date'] = forecast['datetime'].dt.date
    forecast['time'] = forecast['datetime'].dt.time

    true_max_idx = forecast.groupby('date')['total_demand'].idxmax()
    pred_max_idx = forecast.groupby('date')['forecast_demand'].idxmax()
    true_max = forecast.iloc[true_max_idx, 1]
    pred_max = forecast.iloc[pred_max_idx, 2]
    true_max_time = forecast.iloc[true_max_idx, 5]
    pred_max_time = forecast.iloc[pred_max_idx, 5]

    forecast_time = pd.DataFrame(columns=['date', 'true_peak', 'pred_peak', 'true_peak_time', 'pred_peak_time'])
    forecast_time['date'] = forecast['date'].unique()
    forecast_time['true_peak'] = true_max.values
    forecast_time['pred_peak'] = pred_max.values
    forecast_time['true_peak_time'] = true_max_time.values
    forecast_time['pred_peak_time'] = pred_max_time.values
    forecast_time.to_csv(data_folder / 'aemo_peak_and_time.csv', index=False)

