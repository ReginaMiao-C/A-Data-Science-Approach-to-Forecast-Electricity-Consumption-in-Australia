import shap
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import seaborn as sns
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import public_holidays as ph

# ensure reproducibility
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
    

def eval_df(df, all_preds=False):
    """
    return copy of preprocessed data and empty dataframe for response variable evaluation
    all_preds=True returns true vs actual predictions, otherwise returns metrics
    """
    df_datetime = df.copy()
    df_datetime = df[['date', 'time', 'total_demand']]
    if not all_preds:
        results = pd.DataFrame(columns=['date', 'true_peak', 'true_peak_time', 'pred_peak', 'pred_peak_time', 'Total Train MSE', 'Total Train MAE', 'Total Train MAPE', 'Total Val MSE', 'Total Val MAE', 'Total Val MAPE', 'Peak Val MSE', 'Peak Val MAE', 'Peak Val MAPE'])
    else:
        results = pd.DataFrame(columns=['date', 'time', 'total_demand', 'pred_power'])
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


def train_lstm(val_y_start, x, y, criterion_mse, criterion_mae, optimizer, model, train_pass=True):
    """
    train LSTM on 70 days of data before desired validation point
    val_y_start: idx position of first validation datapoint (12am)
    train_pass: True if model is training at current stage
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
    if train_pass:
        model.train()
        for e in range(50):
            output = model(x_train_scaled) 
            mse_loss = criterion_mse(output, y_train_scaled)
            mae_loss = criterion_mae(output, y_train_scaled)
            loss = 0.5 * mse_loss + 0.5 * mae_loss
            optimizer.zero_grad() # reset gradients
            loss.backward() # computes loss gradients
            optimizer.step()

    return scaler_x, scaler_y, val_y_start, x, y, x_train_scaled, model, y_train


def validate_lstm(scaler_x, scaler_y, val_y_start, x, y, x_train_scaled, model, df_datetime):
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
        y_train_pred_scaled = model(x_train_scaled)
        y_pred_scaled = model(x_val_scaled)
    
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.detach().cpu().numpy().T)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy().T)

    day = df_datetime.iloc[val_y_start:val_y_end].copy().reset_index()
    day['pred_power'] = y_pred
    return y_pred, y_train_pred, day, y_val

def calculate_metrics(y_pred, y_train_pred, day, y_val, y_train, results, all_preds):
    """
    evaluate model performance for one window
    """
    if not all_preds:
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
    else:
        day = day.drop(columns='index')
        results = pd.concat([results, day], ignore_index=True)
    return results

def repeat_windows(df, results, df_datetime, initial_val_y_start, num_repeats, days_between_val = 1, retrain=True, all_preds=False):
    """
     repeats sliding window calculations for multiple validation days and returns metrics
     initial_val_y_start: first validation day idx
     num_repeats: number of windows for training and validation
     days_between_val: number of days in 'jumps' between window (if 1, window slides forward by 1 day)
     retrain: specify whether models are retrained at each window
    """
    window_slide = 48*days_between_val
    if initial_val_y_start + (window_slide*(num_repeats-1)) > len(df) - 48:
        print('Error: Validation range cannot exceed ', len(df) - 48)
        print('Current upper validation limit: ', initial_val_y_start + (window_slide*(num_repeats-1)))
        sys.exit()
    *model_info, model = initialise_model(df)
    for r in range(num_repeats):
        val_y_start = initial_val_y_start + (window_slide*r)
        if retrain:
            *train_info, model, y_train = train_lstm(val_y_start, *model_info, model)
        else:
            if r == 0:
                *train_info, model, y_train = train_lstm(val_y_start, *model_info, model)
            else:
                *train_info, model, y_train = train_lstm(val_y_start, *model_info, model, False)
        val_results = validate_lstm(*train_info, model, df_datetime)
        results = calculate_metrics(*val_results, y_train, results, all_preds)

    return results

def display_metrics(results, save=False, file_path='', file_name='', display=True):
    """
    display average metrics for all windows
    save: write all results to csv file
    file_path: folder to save csv file
    file_name: name of csv file
    display: display average metrics over all windows
    """
    if display:
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
        results = results.rename(columns={'total_demand': 'true_demand', 'pred_power': 'lstm_pred_demand'})
        results.to_csv(file_path / file_name)


# wrap model to explain predicted daily peak for SHAP analysis
class PeakModel(nn.Module):
    def __init__(self, base_model):
        """
        base_model: trained LSTM model that outputs 48 half-hour predictions
        """
        super(PeakModel, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        out = self.base_model(x)
        peak_out, _ = torch.max(out, dim=1, keepdim=True)
        return peak_out

def average_shap_training_windows(df, df_datetime, initial_val_y_start, num_repeats, days_between_val=1,
                                  retrain=True, max_background=10, save=False,
                                  file_path='', file_name_prefix='lstm_shap'):
    """
    calculate SHAP values averaged across rolling training windows

    df: preprocessed dataframe with target and feature variables
    initial_val_y_start: first validation day idx
    num_repeats: number of rolling windows
    days_between_val: number of days between windows
    retrain: if True, continue retraining model each window as in repeat_windows
    max_background: number of windows used as SHAP background data
    save: save csv and plots
    file_path: folder to save outputs
    file_name_prefix: prefix for saved files
    """

    window_slide = 48 * days_between_val
    if initial_val_y_start + (window_slide * (num_repeats - 1)) > len(df) - 48:
        print('Error: Validation range cannot exceed ', len(df) - 48)
        print('Current upper validation limit: ', initial_val_y_start + (window_slide * (num_repeats - 1)))
        sys.exit()

    # initialise model and separate x/y data
    x, y, criterion_mse, criterion_mae, optimizer, model = initialise_model(df)
    feature_names = list(x.columns)

    # store scaled training windows and trained model weights
    train_windows = []
    model_states = []

    # create season column (Australian seasons)
    def get_season(date):
        month = pd.to_datetime(date).month
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'

    df_datetime['season'] = df_datetime['date'].apply(get_season)
    
    # training loop
    for r in range(num_repeats):
        val_y_start = initial_val_y_start + (window_slide * r)

        if retrain:
            scaler_x, scaler_y, val_y_start, x, y, x_train_scaled, model, y_train = train_lstm(
                val_y_start, x, y, criterion_mse, criterion_mae, optimizer, model
            )
        else:
            if r == 0:
                scaler_x, scaler_y, val_y_start, x, y, x_train_scaled, model, y_train = train_lstm(
                    val_y_start, x, y, criterion_mse, criterion_mae, optimizer, model
                )
            else:
                scaler_x, scaler_y, val_y_start, x, y, x_train_scaled, model, y_train = train_lstm(
                    val_y_start, x, y, criterion_mse, criterion_mae, optimizer, model, False
                )

        train_windows.append(x_train_scaled.detach().clone())
        model_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # use first few training windows as SHAP background data
    background_data = torch.cat(train_windows[:max_background], dim=0)

    # store aggregated SHAP values for each season
    shap_by_season = {'Summer': [], 'Autumn': [], 'Winter': [], 'Spring': []}
    x_by_season = {'Summer': [], 'Autumn': [], 'Winter': [], 'Spring': []}

    for i in range(len(train_windows)):
        
        val_y_start = initial_val_y_start + (window_slide * i)
        season = df_datetime.iloc[val_y_start]['season']
        
        # reload trained model for this window
        temp_model = LSTMmodel(input_size=x.shape[1], hidden_size=32, num_layers=2, dropout=0)
        temp_model.load_state_dict(model_states[i])
        temp_model.eval()

        peak_model = PeakModel(temp_model)
        explain_data = train_windows[i]

        # create SHAP explainer and calculate values
        explainer = shap.GradientExplainer(peak_model, background_data)
        shap_values = explainer.shap_values(explain_data)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()

        explain_data_np = explain_data.detach().cpu().numpy()

        # aggregate across time steps to get feature-level importance for one window
        shap_values_agg = np.mean(np.abs(shap_values), axis=1).squeeze(0)
        explain_data_agg = np.mean(explain_data_np, axis=1).squeeze(0)

        shap_by_season[season].append(shap_values_agg)
        x_by_season[season].append(explain_data_agg)
    
    seasonal_shap = {}

    for s in shap_by_season:
        if len(shap_by_season[s]) > 0:
            shap_matrix = np.vstack(shap_by_season[s])
            seasonal_shap[s] = shap_matrix.mean(axis=0)
        else:
            seasonal_shap[s] = np.zeros(len(feature_names))
    
    shap_season_df = pd.DataFrame(seasonal_shap, index=feature_names)
    shap_season_df = shap_season_df[['Summer', 'Autumn', 'Winter', 'Spring']]

    plt.figure(figsize=(8,6))
    sns.heatmap(
        shap_season_df,
        cmap='coolwarm',
        annot=True,
        fmt=".2e"
    )
    plt.title('SHAP Feature Importance by Season')
    plt.ylabel('Feature')
    plt.xlabel('Season')
    plt.tight_layout()
    
    plt.savefig(file_path / (file_name_prefix + '_shap_heatmap.png'))
    plt.close()


    # combine all seasons into one matrix
    all_shap = []
    all_x = []
    
    for s in shap_by_season:
        if len(shap_by_season[s]) > 0:
            all_shap.append(np.vstack(shap_by_season[s]))
            all_x.append(np.vstack(x_by_season[s]))
    
    all_shap = np.vstack(all_shap)
    all_x = np.vstack(all_x)
 
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': all_shap.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    if save:
        shap_importance.to_csv(file_path / (file_name_prefix + '_shap_values.csv'), index=False)

        # save bar plot of average SHAP values
        plt.figure(figsize=(8, 6))
        plt.barh(shap_importance['feature'][::-1], shap_importance['mean_abs_shap'][::-1])
        plt.xlabel('mean(|SHAP value|)')
        plt.ylabel('feature')
        plt.tight_layout()
        plt.savefig(file_path / (file_name_prefix + '_shap_bar.png'), bbox_inches='tight')
        plt.close()

        # save SHAP summary plot across all windows
        shap.summary_plot(
            all_shap,
            features=all_x,
            feature_names=feature_names,
            show=False
        )   
        plt.tight_layout()
        plt.savefig(file_path / (file_name_prefix + '_shap_summary.png'), bbox_inches='tight')
        plt.close()

    return shap_importance



### EXAMPLE CODE ###
"""
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

"""