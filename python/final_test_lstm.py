import pandas as pd
from pathlib import Path
import torch
import sys
import numpy as np
import public_holidays as ph
#import test_functions_lstm as lf
import lstm_functions as lf


torch.manual_seed(0)

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

res_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Test'
res_path.mkdir(parents=True,exist_ok=True)

df = pd.read_csv(data_folder / 'all_data_30min.csv')
val_y_start_idx = 3408


df = lf.preprocess_30_min_data(df, True, False, True)

df, df_datetime, results = lf.eval_df(df)

results = lf.repeat_windows(df, results, df_datetime, val_y_start_idx, round((len(df) - val_y_start_idx)/48), 1)
lf.display_metrics(results, True, file_path=res_path, file_name='all_final.csv')