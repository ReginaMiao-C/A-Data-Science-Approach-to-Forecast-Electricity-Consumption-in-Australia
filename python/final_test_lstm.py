import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import public_holidays as ph
import lstm_functions as lf


torch.manual_seed(0)

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data_30min.csv')
val_y_start_idx = 3408

df = lf.preprocess_30_min_data(df, True, False, True)

df, df_datetime, results = lf.eval_df(df)

results = lf.repeat_windows(df, results, df_datetime, val_y_start_idx, round((len(df) - val_y_start_idx)/(48*3)), 3)
lf.display_metrics(results)