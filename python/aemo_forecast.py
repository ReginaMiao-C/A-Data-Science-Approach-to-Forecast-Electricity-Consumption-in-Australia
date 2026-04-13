import pandas as pd
from pathlib import Path
import sys

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'
# set source folder for data
source_folder = Path(r'C:\Users\molly\OneDrive\Documents\UNSW\Project\Data')

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

# get corresponding data to daily peak demand
max_idx = df.groupby('date')['total_demand'].idxmax()
peak_df = df.loc[max_idx].reset_index(drop=True)

# one to many merge on actual peak demand
forecast = pd.merge(peak_df, forecast, on='datetime', how='left')
forecast['forecast_datetime'] = pd.to_datetime(forecast['forecast_datetime'])
forecast['forecast_date'] = forecast['forecast_datetime'].dt.date
forecast['forecast_date'] = pd.to_datetime(forecast['forecast_date'], format='%Y-%m-%d')

# get last forecast from previous day
forecast['prev_date'] = forecast['date'] - pd.Timedelta(days=1)
forecast = forecast[forecast['prev_date'] == forecast['forecast_date']]
forecast = forecast.groupby('datetime').tail(1).reset_index(drop=True)
forecast = forecast.drop(columns=['prev_date', 'forecast_datetime'])

forecast.to_csv(data_folder / 'peak_forecasts.csv', index=False)

