# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load Data
df_sarimax = pd.read_csv('interval_data_with_exog2026-04-24.csv')
df_lstm_train = pd.read_csv('lstm_train.csv')
df_lstm_test = pd.read_csv('lstm_test.csv')
df_AEMO = pd.read_csv('aemo_peak_and_time.csv')

df_AEMO['date'] = pd.to_datetime(df_AEMO['date'])
df_AEMO = df_AEMO.copy()

# Rename the columns
df_AEMO = df_AEMO.rename(columns={
    'pred_peak': 'AEMO_peak_value_predicted',
    'pred_peak_time': 'AEMO_peak_time_predicted',
    'true_peak': 'actual_demand'
})

# Match datetime to the same format as SARIMAX and LSTM
df_AEMO['AEMO_peak_time_predicted'] = pd.to_datetime(
    df_AEMO['date'].astype(str) + ' ' + df_AEMO['AEMO_peak_time_predicted'].astype(str)
)

# Kepp only the 2017 data for training and 2020 for predicting
df_AEMO = df_AEMO[df_AEMO['date'].dt.year.isin([2017, 2020])].copy()

# Split AEMO into train and test datasets
train_df_AEMO = df_AEMO[df_AEMO['date'].dt.year == 2017].copy()
test_df_AEMO = df_AEMO[df_AEMO['date'].dt.year == 2020].copy()

df_sarimax['full_set_mean'] = df_sarimax['full_set_mean'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ')
)

df_sarimax['full_set_index'] = df_sarimax['full_set_index'].apply(
    lambda x: pd.to_datetime(x.replace("'", "").strip('[]').split())
)

df_sarimax['date'] = df_sarimax['full_set_index'].apply(lambda x: x[0]).dt.normalize()
df_sarimax['date'] = pd.to_datetime(df_sarimax['date'])

# Only keep the year 2017(Train) and 2020(Predict)
df_sarimax = df_sarimax[df_sarimax['date'].dt.year.isin([2017, 2020])].copy()

# Get Peak value from full_set_mean, and get corresponding time from full_set_index
peak_vals = []
peak_times = []

for i in range(len(df_sarimax)):
    vals = df_sarimax.iloc[i]['full_set_mean']
    times = df_sarimax.iloc[i]['full_set_index']

    idx = np.argmax(vals)

    peak_vals.append(vals[idx])
    peak_times.append(times[idx])

df_sarimax['sarimax_peak_value_predicted'] = peak_vals
df_sarimax['sarimax_peak_time_predicted'] = peak_times

# Actual Peak from Peak Morning and Peak Afternoon
df_sarimax['sarimax_actual_peak_value'] = df_sarimax[
    ['peak_actual_morning', 'peak_actual_afternoon']
].max(axis=1)

# Find corresponding time for the actual peak
def get_actual_peak_time(row):
    if row['peak_actual_morning'] >= row['peak_actual_afternoon']:
        return row['time_of_peak_actual_morning']
    else:
        return row['time_of_peak_actual_afternoon']

df_sarimax['sarimax_actual_peak_time'] = df_sarimax.apply(get_actual_peak_time, axis=1)

# print(df_sarimax[[
#     'sarimax_peak_value_predicted',
#     'sarimax_peak_time_predicted',
#     'sarimax_actual_peak_value',
#     'sarimax_actual_peak_time'
# ]].head())

# Split data from training and testing
train_df = df_sarimax[df_sarimax['date'].dt.year == 2017].copy()
test_df = df_sarimax[df_sarimax['date'].dt.year == 2020].copy()

# print("Train range:", train_df['date'].min(), "→", train_df['date'].max())
# print("Test range:", test_df['date'].min(), "→", test_df['date'].max())
#
# print("Train size:", len(train_df))
# print("Test size:", len(test_df))

def prepare_lstm(df):
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])
    df['time'] = pd.to_datetime(df['time']).dt.time

    # Combine datetime
    df['lstm_timestamp'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str)
    )

    # get the actual peak and predicted peak
    def get_lstm_peaks(group):
        idx_actual = group['true_demand'].values.argmax()
        idx_pred = group['lstm_pred_demand'].values.argmax()

        return pd.Series({
            'lstm_actual_peak_value': group['true_demand'].iloc[idx_actual],
            'lstm_actual_peak_time': group['lstm_timestamp'].iloc[idx_actual],

            'lstm_peak_value_predicted': group['lstm_pred_demand'].iloc[idx_pred],
            'lstm_peak_time_predicted': group['lstm_timestamp'].iloc[idx_pred]
        })

    return df.groupby('date').apply(get_lstm_peaks).reset_index()

lstm_train_daily = prepare_lstm(df_lstm_train)
lstm_test_daily  = prepare_lstm(df_lstm_test)

# print("LSTM TRAIN range:", lstm_train_daily['date'].min(), "→", lstm_train_daily['date'].max())
# print("LSTM TEST range:", lstm_test_daily['date'].min(), "→", lstm_test_daily['date'].max())
#
# print("LSTM TRAIN size:", len(lstm_train_daily))
# print("LSTM TEST size:", len(lstm_test_daily))

# Merge Training and Testing table
df_train_ensemble = pd.merge(train_df, lstm_train_daily, on='date', how='inner')
df_train_ensemble = pd.merge(
    df_train_ensemble,
    train_df_AEMO[['date', 'AEMO_peak_value_predicted', 'AEMO_peak_time_predicted']],
    on='date',
    how='inner'
)

# Select all the columns
df_train_ensemble = df_train_ensemble[[
    'date',
    'sarimax_peak_value_predicted',
    'sarimax_peak_time_predicted',
    'lstm_peak_value_predicted',
    'lstm_peak_time_predicted',
    'lstm_actual_peak_value',
    'lstm_actual_peak_time',
    'AEMO_peak_value_predicted',
    'AEMO_peak_time_predicted'
]].sort_values('date').reset_index(drop=True)

df_test_ensemble = pd.merge(test_df, lstm_test_daily, on='date', how='inner')
df_test_ensemble = pd.merge(
    df_test_ensemble,
    test_df_AEMO[['date', 'AEMO_peak_value_predicted', 'AEMO_peak_time_predicted']],
    on='date',
    how='inner'
)

# Remove extreme value
df_test_ensemble = df_test_ensemble[
    df_test_ensemble['sarimax_peak_value_predicted'] < 20000
].copy()

df_test_ensemble = df_test_ensemble[[
    'date',
    'sarimax_peak_value_predicted',
    'sarimax_peak_time_predicted',
    'lstm_peak_value_predicted',
    'lstm_peak_time_predicted',
    'lstm_actual_peak_value',
    'lstm_actual_peak_time',
    'AEMO_peak_value_predicted',
    'AEMO_peak_time_predicted'
]].sort_values('date').reset_index(drop=True)

df_train_ensemble = df_train_ensemble.rename(
    columns={
        'lstm_actual_peak_time': 'actual_peak_time',
        'lstm_actual_peak_value': 'actual_peak_value'
    }
)

df_test_ensemble = df_test_ensemble.rename(
    columns={
        'lstm_actual_peak_time': 'actual_peak_time',
        'lstm_actual_peak_value': 'actual_peak_value'
    }
)

print("\nTRAIN ENSEMBLE:")
print(df_train_ensemble.head())

print("\nTEST ENSEMBLE:")
print(df_test_ensemble.head())

print("\nSizes:")
print("Train:", len(df_train_ensemble))
print("Test:", len(df_test_ensemble))

results = []

# Grid search for ensemble weights
for SARIMAX_Weight in np.arange(0, 1.01, 0.01):
    for LSTM_Weight in np.arange(0, 1.01, 0.01):
        AEMO_Weight = 1 - SARIMAX_Weight - LSTM_Weight
        if AEMO_Weight < 0:
            continue
        #AEMO_Weight = 0

        train_pred = (
                SARIMAX_Weight * df_train_ensemble['sarimax_peak_value_predicted'] +
                LSTM_Weight * df_train_ensemble['lstm_peak_value_predicted'] +
                AEMO_Weight * df_train_ensemble['AEMO_peak_value_predicted']
        )

        train_mae = np.mean(
            np.abs(train_pred - df_train_ensemble['actual_peak_value'])
        )

        results.append((SARIMAX_Weight, LSTM_Weight, AEMO_Weight, train_mae))

# Select the best weight
results_df = pd.DataFrame(results, columns=['SARIMAX_weight', 'LSTM_weight', 'AEMO_weight', 'train_MAE'])

best_weight = results_df.loc[results_df['train_MAE'].idxmin()]

SARIMAX_weight, LSTM_weight, AEMO_weight = best_weight[['SARIMAX_weight', 'LSTM_weight', 'AEMO_weight']]

print("Best weight (SARIMAX):", SARIMAX_weight, "| Best weight (LSTM):", LSTM_weight, "| Best weight (AEMO):", AEMO_weight)

# Create Ensemble Prediction
df_test_ensemble['ensemble_peak_value_predicted'] = (
    SARIMAX_weight * df_test_ensemble['sarimax_peak_value_predicted'] +
    LSTM_weight * df_test_ensemble['lstm_peak_value_predicted'] +
    AEMO_weight * df_test_ensemble['AEMO_peak_value_predicted']
)

# Make the peak value into a matrix
train_value_scores = np.vstack([
    df_train_ensemble['sarimax_peak_value_predicted'],
    df_train_ensemble['lstm_peak_value_predicted'],
    df_train_ensemble['AEMO_peak_value_predicted']
])

# Get the time error
train_time_errors = np.vstack([
    (pd.to_datetime(df_train_ensemble['sarimax_peak_time_predicted']) -
     pd.to_datetime(df_train_ensemble['actual_peak_time'])).abs().dt.total_seconds() / 60,

    (pd.to_datetime(df_train_ensemble['lstm_peak_time_predicted']) -
     pd.to_datetime(df_train_ensemble['actual_peak_time'])).abs().dt.total_seconds() / 60,

    (pd.to_datetime(df_train_ensemble['AEMO_peak_time_predicted']) -
     pd.to_datetime(df_train_ensemble['actual_peak_time'])).abs().dt.total_seconds() / 60
])

train_value_norm = train_value_scores / train_value_scores.max(axis=0)
train_time_norm  = train_time_errors / train_time_errors.max(axis=0)

train_score = train_value_norm - train_time_norm

avg_score_per_model = train_score.mean(axis=1)
# Pick the best performing model
best_model_index = np.argmax(avg_score_per_model)

print("Best model index (0=SARIMAX, 1=LSTM, 2=AEMO):", best_model_index)

# Apply best model to testing
df_test_ensemble['ensemble_peak_time_predicted'] = pd.to_datetime(
    np.choose(best_model_index, [
        df_test_ensemble['sarimax_peak_time_predicted'].values,
        df_test_ensemble['lstm_peak_time_predicted'].values,
        df_test_ensemble['AEMO_peak_time_predicted'].values
    ])
)

# Calculate the time error on testing
df_test_ensemble['time_error_minutes'] = (
    df_test_ensemble['ensemble_peak_time_predicted'] -
    pd.to_datetime(df_test_ensemble['actual_peak_time'])
).abs().dt.total_seconds() / 60

print("Avg time error (mins):", df_test_ensemble['time_error_minutes'].mean())

# Compare all three models 

# MAE
# SARIMAX
sarimax_mae = np.mean(
    np.abs(df_test_ensemble['sarimax_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])
)

# LSTM
lstm_mae = np.mean(
    np.abs(df_test_ensemble['lstm_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])
)

AEMO_mae = np.mean(
    np.abs(df_test_ensemble['AEMO_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])
)

# Ensemble
ensemble_mae = np.mean(
    np.abs(df_test_ensemble['ensemble_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])
)

print("SARIMAX MAE:", sarimax_mae)
print("LSTM MAE:", lstm_mae)
print("AEMO MAE:", AEMO_mae)
print("Ensemble MAE:", ensemble_mae)

# MSE
sarimax_mse = np.mean((df_test_ensemble['sarimax_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])**2)
lstm_mse = np.mean((df_test_ensemble['lstm_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])**2)
AEMO_mse = np.mean((df_test_ensemble['AEMO_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])**2)
ensemble_mse = np.mean((df_test_ensemble['ensemble_peak_value_predicted'] - df_test_ensemble['actual_peak_value'])**2)

print("\n--- MSE ---")
print("SARIMAX MSE:", sarimax_mse)
print("LSTM MSE:", lstm_mse)
print("AEMO MSE:", AEMO_mse)
print("Ensemble MSE:", ensemble_mse)

# MAPE
actual = df_test_ensemble['actual_peak_value']

sarimax_mape = np.mean(np.abs((df_test_ensemble['sarimax_peak_value_predicted'] - actual) / actual)) * 100
lstm_mape = np.mean(np.abs((df_test_ensemble['lstm_peak_value_predicted'] - actual) / actual)) * 100
AEMO_mape = np.mean(np.abs((df_test_ensemble['AEMO_peak_value_predicted'] - actual) / actual)) * 100
ensemble_mape = np.mean(np.abs((df_test_ensemble['ensemble_peak_value_predicted'] - actual) / actual)) * 100

print("\n--- MAPE (%) ---")
print("SARIMAX MAPE:", sarimax_mape)
print("LSTM MAPE:", lstm_mape)
print("AEMO MAPE:", AEMO_mape)
print("Ensemble MAPE:", ensemble_mape)

df_test_ensemble.to_csv('test_ensemble_final.csv', index=False)

plt.figure()

plt.plot(
    df_test_ensemble['date'],
    df_test_ensemble['actual_peak_value'],
    label='Actual Peak'
)

plt.plot(
    df_test_ensemble['date'],
    df_test_ensemble['ensemble_peak_value_predicted'],
    label='Ensemble Prediction'
)

plt.title('Ensemble vs Actual Peak Values')
plt.xlabel('Date')
plt.ylabel('Peak Value')
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
