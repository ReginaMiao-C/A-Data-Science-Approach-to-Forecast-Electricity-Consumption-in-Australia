import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


torch.manual_seed(0)

# log norm demand variable
log_norm=False

# changing window size 
num_days = 98
input_window = num_days*48
output_window = 48


# set hyperparameter values
epochs = 50
hidden_size = 128
num_layers = 1
learning_rate = 0.01
dropout_rate = 0.01
weight_decay = 0
mse_weight = 0.75
mae_weight = 0.25
optim_sgd = False

if optim_sgd:
    momentum = 0.9


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data_30min.csv')

## PREPROCESSING ##
# format date and time columns
df['datetime'] = pd.to_datetime(df['datetime'])
df['time'] = df['datetime'].dt.time
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['date'] = df['datetime'].dt.date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df[df['date']!='2010-01-01']

# create dataframe for true deman value comparison
df_datetime = df.copy()
df_datetime = df_datetime.drop(columns=['datetime', 'rainfall', 'pv_capacity', 'temperature', 'solar_power'])
# create results dataframe
results = pd.DataFrame(columns=['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time', 'Train MSE', 'Train MAE', 'Train MAPE', 'Test MSE', 'Test MAE', 'Test MAPE'])


# extract time and data variables
df['year'] = df['date'].dt.year
df['hour'] = df['time'].dt.hour
df['min'] = df['time'].dt.minute
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df = df.drop(columns=['date', 'datetime', 'time'])
df = df.astype('float32')

# cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.where(df['month']==2.0, np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.sin(2 * np.pi * df['day'] / 29), np.sin(2 * np.pi * df['day'] / 28)), np.where(df['month'].isin([4.0, 6.0, 9.0, 11.0]), np.sin(2 * np.pi * df['day'] / 30), np.sin(2 * np.pi * df['day'] / 31)))
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['min_sin'] = np.sin(2 * np.pi * df['min'] / 60)
df['min_cos'] = np.cos(2 * np.pi * df['min'] / 60)

# split feature and response variables
y = df['total_demand']
x = df.drop(columns='total_demand')
if log_norm:
    y = np.log(y)

# create LSTM
class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        input_size: number of feature vars
        hidden_size: number of features in the hidden state (past info stored)
        num_layers: number of recurrent layers (stacked LSTMs)
        dropout: probability of dropout on each layer
        """
        super(LSTMmodel, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = 0)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 48) # output size = 1
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out


num_tests = 100
for repeat in range(num_days, num_days+num_tests):
    print(repeat-num_days)
    # define validation split
    val_start = repeat * 48 
    val_end = val_start + 48

    # min-max scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    # fit on training data
    scaler_x.fit(x[:val_start])
    scaler_y.fit(y[:val_start].values.reshape(-1, 1))
    # scale all data
    x_scaled = scaler_x.transform(x)
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))

    # initialise model
    model = LSTMmodel(input_size=x.shape[1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate)
    criterion_mse = nn.MSELoss() # penalise large errors more heavily
    criterion_mae = nn.L1Loss()
    if optim_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else: 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    y_train_pred_list = []
    y_train_list = []

    train_start = (val_start - input_window) 
    train_end = train_start + input_window

    x_train_scaled = x_scaled[train_start:train_end]
    y_train_scaled = y_scaled[train_end:train_end+output_window]
    x_train_scaled = torch.tensor(x_train_scaled, dtype=torch.float32).unsqueeze(0)
    y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).view(1, -1)

    model.train()
    for e in range(epochs):
        output = model(x_train_scaled) 
        mse_loss = criterion_mse(output, y_train_scaled)
        mae_loss = criterion_mae(output, y_train_scaled)
        loss = mse_weight * mse_loss + mae_weight * mae_loss
        optimizer.zero_grad() # reset gradients
        loss.backward() # computes loss gradients
        optimizer.step() # updates weights from gradients * lr

    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(x_train_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.detach().cpu().numpy().T)
    y_train = scaler_y.inverse_transform(y_train_scaled.detach().cpu().numpy().T)
    if log_norm:
        y_train = np.exp(y_train)
        y_train_pred = np.exp(y_train_pred)
    y_train_pred_list.extend(y_train_pred.tolist())
    y_train_list.extend(y_train.tolist())

    test_start = val_start + 48
    test_end = test_start + output_window
    x_test_scaled = x_scaled[test_start:test_end]
    y_test_scaled = y_scaled[test_end:test_end+output_window]

    x_test_scaled = torch.tensor(x_test_scaled, dtype=torch.float32).unsqueeze(0)
    y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.float32).view(1, -1)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_test_scaled)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy().T)
    if log_norm:
        y_pred = np.exp(y_pred)
    day = df_datetime.iloc[test_end:test_end+output_window].copy().reset_index()      
    day['pred_power'] = y_pred

    true_max = day['total_demand'].max()
    true_max_time= day['time'].iloc[day['total_demand'].idxmax()]
    pred_max = day['pred_power'].max()
    pred_max_time = day['time'].loc[day['pred_power'].idxmax()]

    train_mse = mean_squared_error(y_train_list, y_train_pred_list)
    train_mae = mean_absolute_error(y_train_list, y_train_pred_list)
    train_mape = mean_absolute_percentage_error(y_train_list,y_train_pred_list)*100

    test_mse = mean_squared_error([true_max], [pred_max])
    test_mae = mean_absolute_error([true_max], [pred_max])
    test_mape = mean_absolute_percentage_error([true_max], [pred_max])*100

    # record results for each test
    results.loc[len(results)] = [day['date'].iloc[0], true_max, true_max_time, pred_max, pred_max_time, train_mse, train_mae, train_mape, test_mse, test_mae, test_mape]


# display average results
print('Log norm demand: ', log_norm)
print('Number of days in sliding window: ', num_days)
print('Hidden size: ', hidden_size)
print('Number of layers: ', num_layers)
print('Learning rate: ', learning_rate)
print('Avg train MSE: ', results['Train MSE'].mean())
print('Avg test MSE: ', results['Test MSE'].mean())
print('Avg train MAPE: ', results['Train MAPE'].mean())
print('Avg test MAPE: ', results['Test MAPE'].mean())
print('Avg train MAE: ', results['Train MAE'].mean())
print('Avg test MAE: ', results['Test MAE'].mean())





"""
import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(data=results, x='date', y='true_peak', label='Actual')
sns.lineplot(data=results, x='date', y='pred_peak', label='Predicted')
plt.legend()
plt.show()
plt.close()

fig, axs = plt.subplots(3,1, figsize=(8, 10), layout='constrained')
sns.lineplot(data=results, x='date', y='Train MSE', label='Train', color='blue', ax=axs[0])
sns.lineplot(data=results, x='date', y='Test MSE', label='Test',  color='orange', ax=axs[0])
sns.lineplot(data=results, x='date', y='Train MAE', label='Train',  color='blue', ax=axs[1])
sns.lineplot(data=results, x='date', y='Test MAE', label='Test',  color='orange', ax=axs[1])
sns.lineplot(data=results, x='date', y='Train MAPE', label='Train',  color='blue', ax=axs[2])
sns.lineplot(data=results, x='date', y='Test MAPE', label='Test',  color='orange', ax=axs[2])
axs[0].set_title('MSE')
axs[1].set_title('MAE')
axs[2].set_title('MAPE')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
axs[0].get_legend().remove() 
axs[1].get_legend().remove() 
axs[2].get_legend().remove()
plt.show()
plt.close()"""