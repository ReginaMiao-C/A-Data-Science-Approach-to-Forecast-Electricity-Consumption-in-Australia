import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data_30min.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date
# TODO: remove when original data fixed
df['solar_power'] = df['solar_power'].shift(10)

peak_demand = df.groupby('date')['total_demand'].idxmax()
peak_df = df.loc[df.index.isin(peak_demand)]

demand_stats = df.groupby('date')['total_demand'].agg(['min', 'max', 'mean', 'std'])
peak_df = pd.merge(peak_df, demand_stats.shift(1), on='date')
peak_df = peak_df.rename(columns={'min':'prev_demand_min', 'max':'prev_demand_max', 'mean':'prev_demand_mean', 'std':'prev_demand_std'})

max_temp = df.groupby('date')['temperature'].max()
min_temp = df.groupby('date')['temperature'].min()
temp_stats = df.groupby('date')['temperature'].agg(['min', 'max', 'mean', 'std'])
peak_df = pd.merge(peak_df, temp_stats.shift(1), on='date')
peak_df = peak_df.drop(columns='temperature')
peak_df = pd.merge(peak_df, max_temp, on='date')
peak_df = pd.merge(peak_df, min_temp, on='date')
peak_df = peak_df.rename(columns={'temperature_x':'max_temp', 'temperature_y':'min_temp', 'min':'prev_temp_min', 'max':'prev_temp_max', 'mean':'prev_temp_mean', 'std':'prev_temp_std'})

solar_stats = df.groupby('date')['solar_power'].agg(['min', 'max', 'mean', 'std'])
peak_df = pd.merge(peak_df, solar_stats.shift(1), on='date')
peak_df = peak_df.rename(columns={'min':'prev_solar_min', 'max':'prev_solar_max', 'mean':'prev_solar_mean', 'std':'prev_solar_std', 'total_demand':'peak_demand'})


peak_df['time'] = peak_df['datetime'].dt.time
peak_df['prev_peak_time'] = peak_df['time'].shift(1)
peak_df = peak_df.drop(columns=['time', 'datetime'])
peak_df = peak_df.dropna()

peak_df.to_csv(data_folder / 'all_data_agg.csv')