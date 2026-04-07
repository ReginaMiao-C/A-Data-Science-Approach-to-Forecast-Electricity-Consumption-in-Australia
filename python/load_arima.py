import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def fourier_series(dates, period, K, t0):
    """
    Generates the fourier series for the seasonality of the ARIMAX model, includes both the Sin and Cosine values.
    :param dates: Dates over which to generate the series
    :param period: Period of the fourier series (in minutes)
    :param K: Order of the fourier series (typically 1-3)
    :param t0: Origin of time series
    :return:
    """

    t = (dates - t0).total_seconds() / 60  # minutes since t0
    freq = 2 * np.pi / period
    X = {}
    for k in range(1, K + 1):
        X[f'sin_{period}_{k}'] = np.sin(freq * k * t)
        X[f'cos_{period}_{k}'] = np.cos(freq * k * t)

    return pd.DataFrame(X, index=dates, columns=list(X.keys()))


def load_data():
    cwd = Path.cwd()
    root_folder = cwd.parent
    data_folder = root_folder / "data"

    all_data = pd.read_csv(data_folder / "all_data_30min.csv")
    all_data["datetime"] = pd.to_datetime(all_data["datetime"], yearfirst=True)
    all_data.index = all_data["datetime"]
    all_data.drop("datetime", axis=1, inplace=True)

    # remove the power guard values, fix the data import.
    all_data = all_data.iloc[48:, :]

    # normalise the data:
    # power demand is logged:
    log_demand = np.log1p(all_data["total_demand"])

    # repack data:
    data = pd.DataFrame.from_dict({"total_demand": log_demand})
    data["lag_1"] = data["total_demand"].shift(1)
    data["lag_48"] = data["total_demand"].shift(48)
    data["lag_48*7"] = data["total_demand"].shift(48 * 7)

    # dump the nan rows caused by the shift
    data.dropna(inplace=True, how='any', axis=0)
    weekly_terms = fourier_series(data.index, 7 * 48, K=1, t0=data.index[0])

    complete_dataset = pd.concat([data, weekly_terms], axis=1)

    return complete_dataset


def export_model(pickle_tgt="sarimax.pkl", complete_dataset=None, start_training_data=48*50, training_days=48*7*5):
    # Use the relative paths for dat for this one.

    if complete_dataset is None:
        complete_dataset = load_data()

    end_train = start_training_data + training_days


    train_set = complete_dataset["total_demand"].iloc[start_training_data: end_train]
    train_exog = complete_dataset.drop("total_demand", axis=1).iloc[start_training_data: end_train]

    model = SARIMAX(endog=train_set,
                    exog=train_exog,
                    order=(3, 1, 1),
                    seasonal_order=(1, 0, 1, 48),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                    )

    fit = model.fit(disp=False, maxiter=1000)

    print(fit.summary())

    # dump out the model:
    joblib.dump(model, pickle_tgt, compress=3)
    joblib.dump(fit, "fitted_model.pkl", compress=3)

    return fit


def load_model(pickle_tgt="fitted_model.pkl"):
    try:
        arima_loaded = joblib.load(pickle_tgt)

    except Exception as e:
        print(e)
        arima_loaded = None

    return arima_loaded

if __name__ == "__main__":

    # from the export function:
    start = 48*50
    training_days=48*7*5
    test_set = 48*3

    end =  start + training_days

    data = load_data()

    model = load_model()

    model.predict().plot(color="grey", linewidth=0.5)
    test_data = model.forecast(test_set, exog=data.drop("total_demand", axis=1).iloc[end:end+test_set])
    test_data.plot(color="red", linewidth=0.5, style="--")

    data["total_demand"].iloc[start:end + test_set].plot(color="black", linewidth=2, alpha=0.7)


    plt.show()



