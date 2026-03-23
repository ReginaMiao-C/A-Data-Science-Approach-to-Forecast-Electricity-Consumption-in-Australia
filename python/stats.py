"""
Scripting file to do some statistical analysis of the data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd

# Use the relative paths for dat for this one.
cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / "data"

dispatched_power = pd.read_csv(data_folder / "totaldemand_nsw.csv")
dispatched_power["DATETIME"] = pd.to_datetime(dispatched_power["DATETIME"], dayfirst=True)

# split the datetime into some other units for ease of use:
dispatched_power["YEAR"] = dispatched_power["DATETIME"].dt.year
dispatched_power["MONTH"] = dispatched_power["DATETIME"].dt.month

# don't trust the internal converse
t = (dispatched_power["DATETIME"] - dispatched_power["DATETIME"].min()).dt.total_seconds()

def integrate_group(g):
    t = (g["DATETIME"] - g["DATETIME"].min()).dt.total_seconds()
    return np.trapezoid(g["TOTALDEMAND"], x=t) / (3600*1000)  # GWhr

yearly_energy = dispatched_power[dispatched_power["YEAR"] < dispatched_power["YEAR"].max()].groupby(["YEAR","MONTH"]).apply(integrate_group)

#plt.plot(dispatched_power.groupby("YEAR")["TOTALDEMAND"].mean(), label="Total Demand")
plt.plot(yearly_energy.values, label="Total Demand")

plt.show()
