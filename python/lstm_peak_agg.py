import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


torch.manual_seed(0)
year_month_day = True
cyclical_encoding = True
drop_orig_dates = False

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data_agg.csv')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['prev_peak_time'] = pd.to_datetime(df['prev_peak_time'], format='%H:%M:%S')
df['year'] = df['date'].dt.year

if year_month_day:
    df['day'] = df['date'].dt.strftime('%j') #day of year
else: 
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

# TODO add back in prev peak hour and min
df = df.drop(columns=['date', 'prev_peak_time'])
df = df.astype('float32')

if cyclical_encoding:
    if year_month_day:
        df['day_sin'] = np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.sin(2 * np.pi * df['day'] / 366), np.sin(2 * np.pi * df['day'] / 365))
        df['day_cos'] = np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.cos(2 * np.pi * df['day'] / 366), np.cos(2 * np.pi * df['day'] / 365))
        if drop_orig_dates:
            df = df.drop(columns='day')
    else:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.where(df['month']==2.0, np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.sin(2 * np.pi * df['day'] / 29), np.sin(2 * np.pi * df['day'] / 28)), np.where(df['month'].isin([4.0, 6.0, 9.0, 11.0]), np.sin(2 * np.pi * df['day'] / 30), np.sin(2 * np.pi * df['day'] / 31)))
        if drop_orig_dates:
            df = df.drop(columns=['day', 'month'])

y = df['peak_demand']
x = df.drop(columns='peak_demand')


class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        """
        input_size: number of feature vars
        hidden_size: number of features in the hidden state (past info stored)
        num_layers: number of recurrent layers (stacked LSTMs)
        dropout: probability of dropout on each layer
        """
        super(LSTMmodel, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, 1) # output size = 1

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# sliding window
window_size = 730
forecast = 1
y_pred_list = []
y_test_list = []

for w in range(window_size, window_size+365):
    print(w - window_size)
    x_train = x.iloc[:w]
    x_test = x.iloc[w:w+1]
    y_train = y.iloc[:w]
    y_test = y.iloc[w:w+1]

    scaler_x = MinMaxScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))

    x_train_scaled = torch.tensor(x_train_scaled, dtype=torch.float32).unsqueeze(0)
    y_train_scaled = torch.tensor(y_train_scaled[-1:], dtype=torch.float32)
    x_test_scaled = torch.tensor(x_test_scaled, dtype=torch.float32).unsqueeze(0)
    
    model = LSTMmodel(input_size=x.shape[1], hidden_size=128, num_layers=1)
    criterion_mse = nn.MSELoss() # penalise large errors more heavily
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for e in range(epochs):
        model.train()
        output = model(x_train_scaled) # TODO: consider batching (dataloader)?
        mse_loss = criterion_mse(output, y_train_scaled)
        mae_loss = criterion_mae(output, y_train_scaled)
        loss = 0.5 * mse_loss + 0.5 * mae_loss
        # TODO: experiment with weights?
        optimizer.zero_grad() # reset gradients
        loss.backward() # computes loss gradients
        optimizer.step() # updates weights from gradients * lr

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_test_scaled)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy()).item()
    y_pred_list.append(y_pred)
    y_test_list.append(y_test.values[0])


mse = mean_squared_error(y_test_list, y_pred_list)
mae = mean_absolute_error(y_test_list, y_pred_list)
mape = mean_absolute_percentage_error(y_test_list, y_pred_list)*100

print("Test MSE:", mse)
print("Test MAE:", mae)
print("Test MAPE:", mape)




