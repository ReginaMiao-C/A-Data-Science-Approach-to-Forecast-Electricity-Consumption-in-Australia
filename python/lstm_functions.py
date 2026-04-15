import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import public_holidays as ph

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMmodel, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = 0)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 48) 
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out


def preprocess_30_min_data(df, val_data_only=True, test_data_only=False):
    """
    preprocessing 'all_data_30min.csv' dataframe
    df: 'all_data_30min.csv'
    val_data_only: returns all but 2020 data onwards (ensuring final test data not used)
    test_data_only: returns only testing data
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['time'] = df['datetime'].dt.time
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
    df['date'] = df['datetime'].dt.date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df[df['date']!='2010-01-01'].copy()
    df['year'] = df['date'].dt.year
    df['hour'] = df['time'].dt.hour
    df['min'] = df['time'].dt.minute
    # cyclical encoding
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.where(df['month']==2.0, np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.sin(2 * np.pi * df['day'] / 29), np.sin(2 * np.pi * df['day'] / 28)), np.where(df['month'].isin([4.0, 6.0, 9.0, 11.0]), np.sin(2 * np.pi * df['day'] / 30), np.sin(2 * np.pi * df['day'] / 31)))
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['min_sin'] = np.sin(2 * np.pi * df['min'] / 60)
    df['min_cos'] = np.cos(2 * np.pi * df['min'] / 60)

    if val_data_only:
        df = df[(df['year'] != 2021) & (df['year'] != 2020)]
    elif test_data_only:
        df = df[(df['year'] == 2021) | (df['year'] == 2020)]
    return df
    

def eval_df(df):
    """
    return copy of preprocessed data and empty dataframe for response variable evaluation
    """
    df_datetime = df.copy()
    df_datetime = df[['date', 'time', 'total_demand']]
    results = pd.DataFrame(columns=['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time', 'Total Train MSE', 'Total Train MAE', 'Total Train MAPE', 'Total Val MSE', 'Total Val MAE', 'Total Val MAPE', 'Peak Val MSE', 'Peak Val MAE', 'Peak Val MAPE'])
    df = df.drop(columns=['datetime', 'time', 'date'])
    return df, df_datetime, results

def initialise_model(df):
    """
    separate x and y variables and initialise LSTM
    """
    y = df['total_demand']
    x = df.drop(columns='total_demand')
    model = LSTMmodel(input_size=x.shape[1], hidden_size=32, num_layers=2, dropout=0)
    criterion_mse = nn.MSELoss() # penalise large errors more heavily
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    return x, y, criterion_mse, criterion_mae, optimizer, model


def train_lstm(val_y_start, x, y, criterion_mse, criterion_mae, optimizer, model):
    """
    train LSTM on 70 days of data before desired validation point
    val_y_start: idx position of first validation datapoint (12am)
    """
    if val_y_start < 3408:
        print('Error: Training window precedes start of data. Increase val_y_start to at least 3408')
        sys.exit()
    train_y_end = val_y_start
    train_y_start = train_y_end - 48
    train_x_end = train_y_start
    train_x_start = train_x_end - (70*48)
    x_train = x.iloc[train_x_start:train_x_end]
    y_train = y.iloc[train_y_start:train_y_end]


    #scale data
    scaler_x = MinMaxScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).T
    x_train_scaled = torch.tensor(x_train_scaled, dtype=torch.float32).unsqueeze(0)
    y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)

    #train
    model.train()
    for e in range(50):
        output = model(x_train_scaled) 
        mse_loss = criterion_mse(output, y_train_scaled)
        mae_loss = criterion_mae(output, y_train_scaled)
        loss = 0.5 * mse_loss + 0.5 * mae_loss
        optimizer.zero_grad() # reset gradients
        loss.backward() # computes loss gradients
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(x_train_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.detach().cpu().numpy().T)

    return scaler_x, scaler_y, val_y_start, x, y, model, y_train_pred, y_train


def validate_lstm(scaler_x, scaler_y, val_y_start, x, y, model, df_datetime):
    """
    validate trained model on new y data
    """
    val_y_end = val_y_start + 48
    val_x_end = val_y_start
    val_x_start = val_x_end - (70*48)
    x_val = x.iloc[val_x_start:val_x_end]
    y_val = y.iloc[val_y_start:val_y_end]

    #scale data
    x_val_scaled = scaler_x.transform(x_val)
    x_val_scaled = torch.tensor(x_val_scaled, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy().T)

    day = df_datetime.iloc[val_y_start:val_y_end].copy().reset_index()
    day['pred_power'] = y_pred

    return y_pred, day, y_val

def calculate_metrics(y_pred, day, y_val, y_train, y_train_pred, results):
    """
    evaluate model performance for one window
    """
    true_max = day['total_demand'].max()
    true_max_time= day['time'].iloc[day['total_demand'].idxmax()].time()
    pred_max = day['pred_power'].max()
    pred_max_time = day['time'].loc[day['pred_power'].idxmax()].time()

    total_train_mse = mean_squared_error(y_train, y_train_pred)
    total_train_mae = mean_absolute_error(y_train, y_train_pred)
    total_train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

    total_val_mse = mean_squared_error(y_val, y_pred)
    total_val_mae = mean_absolute_error(y_val, y_pred)
    total_val_mape = mean_absolute_percentage_error(y_val, y_pred)

    peak_val_mse = mean_squared_error([true_max], [pred_max])
    peak_val_mae = mean_absolute_error([true_max], [pred_max])
    peak_val_mape = mean_absolute_percentage_error([true_max], [pred_max])

    results.loc[len(results)] = [day['date'].iloc[0], true_max, true_max_time, pred_max, pred_max_time, total_train_mse, total_train_mae, total_train_mape, total_val_mse, total_val_mae, total_val_mape, peak_val_mse, peak_val_mae, peak_val_mape]

    return results

def repeat_windows(df, results, df_datetime, initial_val_y_start, num_repeats, days_between_val = 1):
    """
     repeats sliding window calculations for multiple validation days and returns metrics
     initial_val_y_start: first validation day idx
     num_repeats: number of windows for training and validation
     days_between_val: number of days in 'jumps' between window (if 1, window slides forward by 1 day)
    """
    window_slide = 48*days_between_val
    if initial_val_y_start + (window_slide*num_repeats) > len(df) - 48:
        print('Error: Validation range cannot exceed ', len(df) - 48)
        sys.exit()
    *model_info, model = initialise_model(df)
    for r in range(num_repeats):
        val_y_start = initial_val_y_start + (window_slide*r)
        *train_info, model, y_train_pred, y_train = train_lstm(val_y_start, *model_info, model)
        val_results = validate_lstm(*train_info, model, df_datetime)
        results = calculate_metrics(*val_results, y_train, y_train_pred, results)

    return results

def display_metrics(results, save=False, file_path='', file_name=''):
    """
    display average metrics for all windows
    save: write all results to csv file
    file_path: folder to save csv file
    file_name: name of csv file
    """
    print('Avg Train MSE: ', results['Total Train MSE'].mean())
    print('Avg Train MAE: ', results['Total Train MAE'].mean())
    print('Avg Train MAPE: ', results['Total Train MAPE'].mean())

    print('Avg Val MSE: ', results['Total Val MSE'].mean())
    print('Avg Val MAE: ', results['Total Val MAE'].mean())
    print('Avg Val MAPE: ', results['Total Val MAPE'].mean())

    print('Avg Peak Val MSE: ', results['Peak Val MSE'].mean())
    print('Avg Peak Val MAE: ', results['Peak Val MAE'].mean())
    print('Avg Peak Val MAPE: ', results['Peak Val MAPE'].mean())

    if save:
        results.to_csv(file_path / file_name)


torch.manual_seed(0)

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data_30min.csv')
val_y_start_idx = 3408

df = preprocess_30_min_data(df)
df, df_datetime, results = eval_df(df)

results = repeat_windows(df, results, df_datetime, val_y_start_idx, 1, 1)
display_metrics(results)