import datetime
from pathlib import Path

from matplotlib import pyplot as plt
from statsforecast.models import AutoARIMA, ARIMA
from statsforecast import StatsForecast
import coreforecast.differences as diff

from fitting import get_stats, get_data_normalised
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf

if __name__=="__main__":



    cwd = Path.cwd()
    root_folder = cwd.parent.parent
    data_folder = root_folder / "data"
    data = pd.read_csv(data_folder / "all_data_30min.csv")
    energy = data["total_demand"]
    seasonal = energy.diff(periods=48)
    seasonal.dropna(inplace=True)

    res = diff.num_diffs(energy.values, max_d=5)
    print(res)

    res = diff.num_diffs(seasonal.values[0:48*8], max_d=5)
    print(res)


    # Plot ACF/PACF after differencing
    #fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    #plot_acf(seasonal[0:48*10], ax=axes[0], lags=96)
    #axes[0].set_title('ACF (Differenced)')

    #plot_pacf(seasonal[0:48*10], ax=axes[1], lags=96, method='ywm')
    #axes[1].set_title('PACF (Differenced)')

    #plt.tight_layout()
    #plt.show()


    #plt.show()




