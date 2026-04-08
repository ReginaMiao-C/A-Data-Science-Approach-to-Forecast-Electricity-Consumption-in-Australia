
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt
from statsmodels.tsa.api import SARIMAX


cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"

df = pd.read_csv(data_folder / "all_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['dayofyear'] = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
df['temp_range'] = df['max_temperature'] - df['min_temperature']
df['rain_solar_interaction'] = df['rainfall'] * df['solar_exposure']
df['lag1'] = df['peak_power'].shift(1)
df['lag7'] = df['peak_power'].shift(7)
df['lag14'] = df['peak_power'].shift(14)

df = df.dropna().reset_index(drop=True)

sarimax_model = SARIMAX(
    df['peak_power'],
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarimax_fit = sarimax_model.fit(disp=False, maxiter=500)
sarimax_pred = sarimax_fit.fittedvalues

residuals = df['peak_power'] - sarimax_pred

df['resid_lag1'] = residuals.shift(1)
df['resid_lag7'] = residuals.shift(7)

df = df.dropna().reset_index(drop=True)

feature_cols = [
    'sin_day', 'cos_day', 'dayofyear',
    'min_temperature', 'max_temperature',
    'solar_exposure', 'rainfall', 'pv_capacity',
    'lag1', 'lag7', 'lag14',
    'resid_lag1', 'resid_lag7', 'temp_range', 'rain_solar_interaction'
]
X = df[feature_cols]
y = df['peak_power'] - sarimax_fit.fittedvalues.loc[df.index]

split_point = int(len(df) * 0.8)
X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]

sarimax_test_pred = sarimax_fit.forecast(steps=len(X_test))

estimators_options = [300, 350, 400, 500]
depth_options = [3, 4]
learning_rates = [0.01, 0.03, 0.05]

best_mse = float('inf')
best_pred = None
best_combo = None
best_mae = None
best_mape = None

for n_est in estimators_options:
    for max_d in depth_options:
        for lr in learning_rates:
            gbr = GradientBoostingRegressor(
                n_estimators=n_est,
                max_depth=max_d,
                learning_rate=lr,
                max_features = 'sqrt',
                random_state=0
            )
            gbr.fit(X_train, y_train)
            resid_pred = gbr.predict(X_test)

            hybrid_pred = sarimax_test_pred.to_numpy() + resid_pred
            y_true = df['peak_power'].iloc[split_point:].values

            mse = mean_squared_error(y_true, hybrid_pred)
            mae = mean_absolute_error(y_true, hybrid_pred)
            mape = mean_absolute_percentage_error(y_true, hybrid_pred) * 100

            if mse < best_mse:
                best_mse = mse
                best_mae = mae
                best_mape = mape
                best_combo = (n_est, max_d, lr)
                best_pred = hybrid_pred.copy()

            print(f"Combo: n_est={n_est}, max_depth={max_d}, lr={lr}")
            print(f"MSE: {mse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%\n")


plt.figure()
plt.plot(df['date'].iloc[split_point:], y_true, label='Actual')
plt.plot(df['date'].iloc[split_point:], best_pred, label=f'Best Prediction {best_combo}', linestyle = '--')
plt.legend()
plt.title('Best Hybrid SARIMAX And Gradient Boosting Prediction')
plt.xlabel('Date')
plt.ylabel('Peak Power')
plt.show()

print(f"Best combo: n_est={best_combo[0]}, max_depth={best_combo[1]}, lr={best_combo[2]}")
print(f"Lowest MSE: {best_mse:.2f}| Lowest MAE: {best_mae:.2f} | Lowest MAPE: {best_mape:.2f}%\n")
