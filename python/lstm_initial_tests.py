import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


torch.manual_seed(0)
use_all_data = True
year_month_day = False
cyclical_encoding = True
drop_orig_dates = False

cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data.csv')
df = df.rename(columns={'Unnamed: 0': 'date'})
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

if year_month_day:
    df['day'] = df['date'].dt.strftime('%j') #day of year
else: 
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
df = df.drop(columns='date')
df = df.astype('float32')

# cyclical feature encoding
# https://machinelearningmastery.com/7-pandas-tricks-for-time-series-feature-engineering/
if cyclical_encoding: # TODO: confirm how to handle leap years
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



# split into training and testing
if use_all_data:
    test = df.iloc[-1:]
    train = df.iloc[:-1]
else:
    train = df[df['year']== 2010]
    test = df[(df['year'] == 2011)].head(1)



y_train = train['peak_power']
y_test = test['peak_power']
x_train = train.drop(columns='peak_power')
x_test = test.drop(columns='peak_power')


# values for LSTM input
seq_length = len(x_train)
feature_vars = len(x_train.columns)


# scale
scaler_x = MinMaxScaler()
scaler_x.fit(x_train)
x_train_scaled = scaler_x.transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)) #2d array

# transorm into tensors
y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
x_train = torch.tensor(x_train_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
x_test = torch.tensor(x_test_scaled, dtype=torch.float32)


# NOTE: fixed dates, will remove later
x_train = x_train.unsqueeze(0) 
y_train = y_train[-1:] 
x_test = x_test.unsqueeze(0) 
y_test = y_test[-1:] 


# build LSTM
# https://www.geeksforgeeks.org/data-analysis/time-series-forecasting-using-pytorch/
# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
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
    

# train model
model = LSTMmodel(input_size=feature_vars, hidden_size=64, num_layers=2)
criterion_mse = nn.MSELoss() # penalise large errors more heavily
criterion_mae = nn.L1Loss()
# NOTE: could consider Huber loss as an alternative
# TODO: add multiple loss functions (MAE as well)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# TODO: consider other optimisers/learning rate

epochs = 50
for e in range(epochs):
    model.train()
    output = model(x_train) # TODO: consider batching (dataloader)?
    mse_loss = criterion_mse(output.squeeze(), y_train)
    mae_loss = criterion_mae(output.squeeze(), y_train)
    loss = 0.5 * mse_loss + 0.5 * mae_loss
    # TODO: experiment with weights?
    optimizer.zero_grad() # reset gradients
    loss.backward() # computes loss gradients
    optimizer.step() # updates weights from gradients * lr
    print(f'Epoch {e+1}, Loss: {loss.item():.4f}')


# evaluate model
model.eval()

with torch.no_grad():
    y_pred_scaled = model(x_test).squeeze()

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

print(f'True value: {round(y_test.item(), 3)}\nPredicted value: {round(y_pred.item(), 3)}\n')
print(f'Test MAE: {round(mean_squared_error(y_test, y_pred), 3)}')
print(f'Test MSE: {round(mean_absolute_error(y_test, y_pred), 3)}')
print(f'Test MAPE: {round(mean_absolute_percentage_error(y_test, y_pred), 3)*100}%')


