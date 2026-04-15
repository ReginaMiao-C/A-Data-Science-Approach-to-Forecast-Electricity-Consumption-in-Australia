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

# ensure reproducible results
torch.manual_seed(0)

day_of_year = False
drop_orig_dates = False #cyclical only
ymd = ''
cyclical_only = ''
holidays = False
no_ph = ''

keep_weights = True

# specify vars for droupout tests
rainfall = True
pv = True
temp = True
solar = True
holidays = True

# define which model used for validation
model_1 = True
model_2 = False


epochs = 50

if model_1:
    window_size = 70
    hidden_size = 32
    num_layers = 2
    mse_weight = 0.5
    mae_weight = 0.5
    learning_rate = 0.005
    dropout_rate = 0
    model_name = 'model1'
if model_2:
    window_size = 98
    hidden_size = 128
    num_layers = 1
    mse_weight = 1
    mae_weight = 0
    learning_rate = 0.01
    dropout_rate = 0
    model_name = 'model2'




cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'

df = pd.read_csv(data_folder / 'all_data_30min.csv')


## PREPROCESSING ##

df['datetime'] = pd.to_datetime(df['datetime'])
df['time'] = df['datetime'].dt.time
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['date'] = df['datetime'].dt.date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df[df['date']!='2010-01-01']


df['year'] = df['date'].dt.year
df['hour'] = df['time'].dt.hour
df['min'] = df['time'].dt.minute


if holidays:
    no_ph = 'ph'
    public_hols = []
    for yr in df['year'].unique():
        public_hols.extend(ph.get_holidays(yr))
    public_hol_dates = pd.to_datetime(public_hols)
    df['public_hol'] = df['date'].isin(public_hol_dates).astype(int)


# cyclical encoding
if day_of_year:
    ymd = 'ymd_'
    df['day'] = df['date'].dt.strftime('%j').astype(float) #day of year
    df['day_sin'] = np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.sin(2 * np.pi * df['day'] / 366), np.sin(2 * np.pi * df['day'] / 365))
    df['day_cos'] = np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.cos(2 * np.pi * df['day'] / 366), np.cos(2 * np.pi * df['day'] / 365))
    if drop_orig_dates:
        cyclical_only = 'cyclical_only_'
        df = df.drop(columns='day')
else: 
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.where(df['month']==2.0, np.where(df['year'].isin([2012.0, 2016.0, 2020.0]), np.sin(2 * np.pi * df['day'] / 29), np.sin(2 * np.pi * df['day'] / 28)), np.where(df['month'].isin([4.0, 6.0, 9.0, 11.0]), np.sin(2 * np.pi * df['day'] / 30), np.sin(2 * np.pi * df['day'] / 31)))
    if drop_orig_dates:
        df = df.drop(columns=['day', 'month'])
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['min_sin'] = np.sin(2 * np.pi * df['min'] / 60)
df['min_cos'] = np.cos(2 * np.pi * df['min'] / 60)
if drop_orig_dates:
    df = df.drop(columns=['hour', 'min'])

dropped_vars = '_'
# remove vars
if not rainfall:
    df = df.drop(columns='rainfall')
    dropped_vars = dropped_vars + 'rainfall_'
if not pv:
    df = df.drop(columns='pv_capacity')
    dropped_vars = dropped_vars + 'pv_'
if not temp:
    df = df.drop(columns='temperature')
    dropped_vars = dropped_vars + 'temperature_'
if not solar:
    df = df.drop(columns='solar_power')
    dropped_vars = dropped_vars + 'solar_'

# separate unseen final testing data
test_data = df[(df['year'] == 2021) | (df['year'] == 2020)]
val_data = df[(df['year'] != 2021) & (df['year'] != 2020)]
val_data = val_data.reset_index()
# create copy of peak data for evaluation
df_datetime = val_data.copy()
df_datetime = val_data[['date', 'time', 'total_demand']]
val_data = val_data.drop(columns=['datetime', 'time', 'date'])

# separate x and y data
y = val_data['total_demand']
x = val_data.drop(columns='total_demand')


# define model
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
        self.linear = nn.Linear(hidden_size, 48) 

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.linear(out)
        return out

# create dataframe to store results
results = pd.DataFrame(columns=['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time', 'Total Train MSE', 'Total Train MAE', 'Total Train MAPE', 'Total Val MSE', 'Total Val MAE', 'Total Val MAPE', 'Peak Val MSE', 'Peak Val MAE', 'Peak Val MAPE'])

if keep_weights:
    # initialise model
    model = LSTMmodel(input_size=x.shape[1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate)
    criterion_mse = nn.MSELoss() # penalise large errors more heavily
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
epoch_list = []

# split data for each training window
obs = 48 #number of observations in a day
# val sample every 40 days to ensure good spread of days and months sampled across all years
for repeat in range(98*obs, len(val_data), 40*obs):
    print((repeat-98*obs)/(40*obs))
    # train on 70 day window to predict next day
    # validate on following day
    train_y_start = repeat
    train_y_end = repeat + obs
    train_x_start = repeat - window_size*obs
    train_x_end = train_y_start
    val_y_start = train_y_start + obs
    val_y_end = train_y_end + obs
    val_x_start = train_x_start + obs
    val_x_end = val_y_start

    x_train = x.iloc[train_x_start:train_x_end]
    y_train = y.iloc[train_y_start:train_y_end]
    x_val = x.iloc[val_x_start:val_x_end]
    y_val = y.iloc[val_y_start:val_y_end]
    
    # scale on training data
    scaler_x = MinMaxScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_val_scaled = scaler_x.transform(x_val)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).T
    #y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).T
    

    x_train_scaled = torch.tensor(x_train_scaled, dtype=torch.float32).unsqueeze(0)
    y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32)
    x_val_scaled = torch.tensor(x_val_scaled, dtype=torch.float32).unsqueeze(0)
    #y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.float32)



    if not keep_weights:
        # initialise model
        model = LSTMmodel(input_size=x.shape[1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate)
        criterion_mse = nn.MSELoss() # penalise large errors more heavily
        criterion_mae = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    # train model
    model.train()
    for e in range(epochs):
        output = model(x_train_scaled) 
        mse_loss = criterion_mse(output, y_train_scaled)
        mae_loss = criterion_mae(output, y_train_scaled)
        loss = mse_weight * mse_loss + mae_weight * mae_loss
        optimizer.zero_grad() # reset gradients
        loss.backward() # computes loss gradients
        optimizer.step()
        train_loss.append(loss.item())
        epoch_list.append(e)

    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(x_train_scaled)
        y_pred_scaled = model(x_val_scaled)

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.detach().cpu().numpy().T)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy().T)
    
    # get true values and times for validation day
    day = df_datetime.iloc[val_y_start:val_y_end].copy().reset_index()
    day['pred_power'] = y_pred
 

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
    

res_path = root_folder / 'Results' / 'LSTM'
res_path.mkdir(parents=True,exist_ok=True)
if keep_weights:
    csv_name = 'train_val_results_keep_weights_' + model_name + dropped_vars + ymd + cyclical_only + no_ph +'.csv'
    plot_name = 'peak_results_keep_weights_' + model_name + dropped_vars + ymd + cyclical_only + no_ph
    plot_name_loss = 'train_loss_keep_weights_' + model_name + dropped_vars + ymd + cyclical_only + no_ph
    results.to_csv(res_path / csv_name)
if not keep_weights:
    csv_name = 'train_val_results_' + model_name + dropped_vars + ymd + cyclical_only + no_ph + '.csv'
    plot_name = 'peak_results_' + model_name + dropped_vars + ymd + cyclical_only + no_ph
    plot_name_loss = 'train_loss_' + model_name + dropped_vars + ymd + cyclical_only + no_ph
    results.to_csv(res_path / csv_name)

sns.lineplot(data=results, x='date', y='true_peak', label='Actual')
sns.lineplot(data=results, x='date', y='pred_peak', label='Predicted')
plt.legend()
plt.savefig(res_path / plot_name)
plt.close()


loss_df = pd.DataFrame({'train_loss': train_loss, 'epoch': epoch_list}).reset_index()

sns.lineplot(loss_df, y='train_loss', x='index')
plt.savefig(res_path / plot_name_loss)
plt.close()
plot_name_loss = 'epoch_' + plot_name_loss
sns.lineplot(loss_df, y='train_loss', x='epoch')
plt.savefig(res_path / plot_name_loss)
plt.close()

print('Avg Train MSE: ', results['Total Train MSE'].mean())
print('Avg Train MAE: ', results['Total Train MAE'].mean())
print('Avg Train MAPE: ', results['Total Train MAPE'].mean())

print('Avg Val MSE: ', results['Total Val MSE'].mean())
print('Avg Val MAE: ', results['Total Val MAE'].mean())
print('Avg Val MAPE: ', results['Total Val MAPE'].mean())

print('Avg Peak Val MSE: ', results['Peak Val MSE'].mean())
print('Avg Peak Val MAE: ', results['Peak Val MAE'].mean())
print('Avg Peak Val MAPE: ', results['Peak Val MAPE'].mean())