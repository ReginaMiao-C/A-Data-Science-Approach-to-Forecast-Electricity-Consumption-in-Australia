import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import public_holidays as ph

torch.manual_seed(0)

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


def preprocess_30_min_data(df, holidays = True, val_data_only=True, test_data_only=False):
    """
    preprocessing 'all_data_30min.csv' dataframe
    df: 'all_data_30min.csv'
    holidays: include public holiday data
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

    if holidays:
        public_hols = []
        for yr in df['year'].unique():
            public_hols.extend(ph.get_holidays(yr))
        public_hol_dates = pd.to_datetime(public_hols)
        df['public_hol'] = df['date'].isin(public_hol_dates).astype(int)


    # cyclical encoding
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.where(df['month']==2.0, np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.sin(2 * np.pi * df['day'] / 29), np.sin(2 * np.pi * df['day'] / 28)), np.where(df['month'].isin([4.0, 6.0, 9.0, 11.0]), np.sin(2 * np.pi * df['day'] / 30), np.sin(2 * np.pi * df['day'] / 31)))
    df['day_cos'] = np.where(df['month']==2.0, np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.cos(2 * np.pi * df['day'] / 29), np.cos(2 * np.pi * df['day'] / 28)), np.where(df['month'].isin([4.0, 6.0, 9.0, 11.0]), np.sin(2 * np.pi * df['day'] / 30), np.cos(2 * np.pi * df['day'] / 31)))
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['min_sin'] = np.sin(2 * np.pi * df['min'] / 60)
    df['min_cos'] = np.cos(2 * np.pi * df['min'] / 60)

    if val_data_only:
        df = df[(df['year'] != 2021) & (df['year'] != 2020)]
    elif test_data_only:
        df = df[(df['year'] == 2020)]
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

def split_data(initial_val_y_start, num_repeats, window_slide, x, y, df_datetime):
    days = pd.DataFrame(columns=['date', 'time', 'total_demand'])
    x_train_seq = []
    y_train_seq = []
    x_val_seq = []
    y_val_seq = []
    # create sequences of training and validation data (validation size of 50)
    for r in range(num_repeats-50):
        val_y_start = initial_val_y_start + (window_slide*r)
        train_y_end = val_y_start
        train_y_start = train_y_end - 48
        train_x_end = train_y_start
        train_x_start = train_x_end - (70*48)
        x_train = x.iloc[train_x_start:train_x_end].to_numpy()
        y_train = y.iloc[train_y_start:train_y_end].to_numpy()
        x_train_seq.append(x_train)
        y_train_seq.append(y_train)
    for r in range(num_repeats-50, num_repeats):
        val_y_start = initial_val_y_start + (window_slide*r)
        val_y_end = val_y_start + 48
        val_x_end = val_y_start
        val_x_start = val_x_end - (70*48)
        x_val = x.iloc[val_x_start:val_x_end].to_numpy()
        y_val = y.iloc[val_y_start:val_y_end].to_numpy()
        x_val_seq.append(x_val)
        y_val_seq.append(y_val)
        day = df_datetime.iloc[val_y_start:val_y_end].copy().reset_index(drop=True)
        days = pd.concat([days, day], ignore_index=True)

    x_train_seq = np.array(x_train_seq)
    y_train_seq = np.array(y_train_seq)
    x_val_seq = np.array(x_val_seq)
    y_val_seq = np.array(y_val_seq)
    return x_train_seq, y_train_seq, x_val_seq, y_val_seq, days



def repeat_windows(df, results, df_datetime, initial_val_y_start, num_repeats, days_between_val = 1):
    """
     repeats sliding window calculations for multiple validation days and returns metrics
     initial_val_y_start: first validation day idx
     num_repeats: number of windows for training and validation
     days_between_val: number of days in 'jumps' between window (if 1, window slides forward by 1 day)
    """
    window_slide = 48*days_between_val
    if initial_val_y_start + (window_slide*(num_repeats-1)) > len(df) - 48:
        print('Error: Validation range cannot exceed ', len(df) - 48)
        print('Current upper validation limit: ', initial_val_y_start + (window_slide*(num_repeats-1)))
        sys.exit()
    x, y, criterion_mse, criterion_mae, optimizer, model = initialise_model(df)

    x_train, y_train, x_val, y_val, days = split_data(initial_val_y_start, num_repeats, window_slide, x, y, df_datetime)

    scaler_x = MinMaxScaler()
    x_train_flat = x_train.reshape(-1, x_train.shape[-1])
    x_val_flat = x_val.reshape(-1, x_val.shape[-1])

    x_train_scaled = scaler_x.fit_transform(x_train_flat).reshape(x_train.shape)
    x_val_scaled = scaler_x.transform(x_val_flat).reshape(x_val.shape)
    x_train_scaled = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_val_scaled = torch.tensor(x_val_scaled, dtype=torch.float32)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)

 

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
        y_pred_scaled = model(x_val_scaled)

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.detach().cpu().numpy()).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy()).flatten()

    days['pred_power'] = y_pred
    true_peak_days_idx = days.groupby(days['date'])['total_demand'].idxmax()
    pred_peak_days_idx = days.groupby(days['date'])['pred_power'].idxmax()
    
    true_peak_days = days.loc[true_peak_days_idx]
    pred_peak_days = days.loc[pred_peak_days_idx]
    # df[['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time']]
    true_peak_days = true_peak_days[['date', 'time', 'total_demand']]
    true_peak_days = true_peak_days.rename(columns={'time': 'true_peak_time', 'total_demand': 'true_peak'})
    pred_peak_days = pred_peak_days[['date', 'time', 'pred_power']]
    pred_peak_days = pred_peak_days.rename(columns={'time': 'pred_peak_time', 'pred_power': 'pred_peak'})

    res = pd.merge(true_peak_days, pred_peak_days, on='date')
    print(res.head())
    print('Avg Train MSE: ', mean_squared_error(y_train.flatten(), y_train_pred))
    print('Avg Train MAE: ', mean_absolute_error(y_train.flatten(), y_train_pred))
    print('Avg Train MAPE: ', mean_absolute_percentage_error(y_train.flatten(), y_train_pred))

    print('Avg Val MSE: ', mean_squared_error(y_val.flatten(), y_pred))
    print('Avg Val MAE: ', mean_absolute_error(y_val.flatten(), y_pred))
    print('Avg Val MAPE: ', mean_absolute_percentage_error(y_val.flatten(), y_pred))

    print('Avg Peak Val MSE: ', mean_squared_error(true_peak_days['true_peak'], pred_peak_days['pred_peak']))
    print('Avg Peak Val MAE: ', mean_absolute_error(true_peak_days['true_peak'], pred_peak_days['pred_peak']))
    print('Avg Peak Val MAPE: ', mean_absolute_percentage_error(true_peak_days['true_peak'], pred_peak_days['pred_peak']))

    return res

torch.manual_seed(0)

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data_30min.csv')
val_y_start_idx = 3408

df = preprocess_30_min_data(df)
df, df_datetime, results = eval_df(df)

results = repeat_windows(df, results, df_datetime, val_y_start_idx, round((len(df) - val_y_start_idx)/(48*20)), 20)