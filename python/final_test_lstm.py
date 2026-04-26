import pandas as pd
from pathlib import Path
import torch
import sys
import numpy as np
import public_holidays as ph
import lstm_functions as lf

# ensure reproducible results
torch.manual_seed(0)

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'
df = pd.read_csv(data_folder / 'all_data_30min.csv')
# testing: 1 = actual best model (no datetime vars), 2 = predictions for ensemble, 3 = best model (all vars), 4 = sequential model - proof of concept
use = 1

if use == 1:
    res_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Var Dropout'
    val_y_start_idx = 3408 

    df = lf.preprocess_30_min_data(df, True, False, True)
    df, df_datetime, results = lf.eval_df(df)
    df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand', 'public_hol']]
    results = lf.repeat_windows(df, results, df_datetime, val_y_start_idx, round((len(df) - val_y_start_idx)/48), 1)
    lf.display_metrics(results, True, file_path=res_path, file_name='dropped_8_test.csv')

elif use == 2:
    res_path = cwd / 'ensemble_data'
    # training data
    val_y_start_idx = 122688 #starts at 2017

    df = lf.preprocess_30_min_data(df, True)
    df, df_datetime, results = lf.eval_df(df, all_preds=True)
    df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand', 'public_hol']]
    results = lf.repeat_windows(df, results, df_datetime, val_y_start_idx, 365, 1, retrain=True, all_preds=True)
    lf.display_metrics(results, True, file_path=res_path, file_name='lstm_train.csv', display=False)

    #testing data
    val_y_start_idx = 3408 
    df = pd.read_csv(data_folder / 'all_data_30min.csv')
    df = lf.preprocess_30_min_data(df, True, False, True)
    df, df_datetime, results = lf.eval_df(df, all_preds=True)
    df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand', 'public_hol']]
    results = lf.repeat_windows(df, results, df_datetime, val_y_start_idx,  round((len(df) - val_y_start_idx)/48), 1, retrain=True, all_preds=True)
    lf.display_metrics(results, True, file_path=res_path, file_name='lstm_test.csv', display=False)

elif use == 3:
    res_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Test'
    res_path.mkdir(parents=True,exist_ok=True)
    val_y_start_idx = 3408 

    df = lf.preprocess_30_min_data(df, True, False, True)
    df, df_datetime, results = lf.eval_df(df)
    results = lf.repeat_windows(df, results, df_datetime, val_y_start_idx, round((len(df) - val_y_start_idx)/48), 1)
    lf.display_metrics(results, True, file_path=res_path, file_name='all_final.csv')

elif use == 4:
    import lstm_functions_multi_seq as lf
    val_y_start_idx = 3408

    df = lf.preprocess_30_min_data(df, True, False, True)
    df, df_datetime, results = lf.eval_df(df)
    results = lf.repeat_windows(df, results, df_datetime, val_y_start_idx, round((len(df) - val_y_start_idx)/48), 1)
    file_path=root_folder / 'Results' / 'LSTM' / 'Final' / 'Test'
    results.to_csv(file_path / 'multi_seq_test.csv')
