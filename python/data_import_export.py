"""
Python script to handle importing, cleaning and data export.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata


import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def repair_dataframe(df, value_col, datetime_col="DATE", method="linear"):
    """
    Fill NaN values using 2D interpolation across (year, position in year).

    Args:
        df : incoming dataframe
        value_col:  Column with values to interpolate
        datetime_col : the name of the time column
        method : interpolation method
    Returns:
        pd.DataFrame Copy of df with NaNs filled in value_col
    """

    df_out = df.copy()

    dt = pd.to_datetime(df_out[datetime_col])

    # will be pivoting on the year, so need to split the date by the yar.
    year = dt.dt.year
    start_of_year = pd.to_datetime(year.astype(str) + "-01-01")
    next_year = pd.to_datetime((year + 1).astype(str) + "-01-01")
    t_in_year = (dt - start_of_year) / (next_year - start_of_year)

    mask = df_out[value_col].notna()

    points = np.column_stack((year[mask], t_in_year[mask]))
    values = df_out.loc[mask, value_col]

    points_missing = np.column_stack((year[~mask], t_in_year[~mask]))

    if len(points_missing) > 0:
        df_out.loc[~mask, value_col] = griddata(points, values, points_missing, method=method)

    return df_out

def import_data(cwd:Path, filename:str, rows = None):
    """
    function to import data
    Args:
        cwd: current working directory as a Path object
        filename: filename of the csv file as a string
        rows: number of rows to skip for the header
    Returns:
        pd.DataFrame
    Raises:
        FileNotFoundError: if filename does not exist
        Exception: if more than one file exists
    """
    files = list(cwd.rglob(f"*{filename}*"))

    if len(files) == 0:
        raise FileNotFoundError(f"File {filename} not found.")
    elif len(files) > 1:
        raise Exception(f"More than one file {filename} found.")
    else:
        file = files[0]
        # files have different headers so just loop until we find the data, as they all do.

        if rows is None:
            for i in range(20):
                try:
                    df = pd.read_csv(file, skiprows=i)
                except Exception as e:
                    continue

                break
        else:
            df = pd.read_csv(file, skiprows=rows)

    return df

def set_date(dataframe, fmt=None, date_col="Date"):
    """
    Args:
    dataframe: pandas dataframe of the data to be modified
    fmt:  format of the date string.
    date_col: column name of the date, default is 'Date'
    """

    dataframe.set_index(dataframe[date_col], inplace=True)

    return dataframe

def interpolate_data(date_range, data, column):

    # repair the data from using a 2D interpolation over the available time
    temp_df = repair_dataframe(data, column)

    # set the range to seconds from the epoch.
    dt_min  = date_range.min()
    index = temp_df["DATE"]
    range_seconds = (date_range - dt_min).dt.total_seconds()
    seconds = (index - dt_min).dt.total_seconds()

    # interpolate with some guard values for checking.
    interpolated_data =  np.interp(range_seconds, seconds, temp_df[column], left=-999, right=-999)

    return interpolated_data



if __name__ == "__main__":

    cwd = Path.cwd()
    cwd = cwd.parent / "data"

    # load the data and create the datetime as necessary:
    solar_power =  import_data(cwd, "POWER_POINT",10)
    solar_power["DATE"] = pd.to_datetime({"year": solar_power["YEAR"], "month": solar_power["MO"], "day": solar_power["DY"],"hour": solar_power["HR"],})

    # solar power is at UTC, Sydney is at UTC+10
    solar_power["DATE"] = solar_power["DATE"] + pd.Timedelta(hours=10)

    demand = import_data(cwd, "totaldemand_nsw")
    demand["DATE"] = pd.to_datetime(demand["DATETIME"], dayfirst=True)

    temperature = import_data(cwd, "temperature_nsw")
    temperature["DATE"] = pd.to_datetime(temperature["DATETIME"], dayfirst=True)

    rainfall = import_data(cwd, "daily_rainfall_bankstown_data")
    rainfall["DATE"] = pd.to_datetime({"year": rainfall["Year"], "month": rainfall["Month"], "day": rainfall["Day"]})

    # interpolate and repair data to the demand timesteps:
    temperature_interpolated = interpolate_data(demand["DATE"], temperature, "TEMPERATURE")
    solar_power_interpolated = interpolate_data(demand["DATE"], solar_power, "ALLSKY_SFC_SW_DWN")

    # PV and rainfall is monthly and daily data, so handle differently
    pv_data = import_data(cwd, "monthly_pv_installations_nsw.csv")
    pv_data.index = pv_data["DATE"]

    # PV data is monthly integer values, so won't interpolate down for this one, slow but fine for a one shot.
    pv_key = demand.apply(lambda x: f"{str(x["DATE"].year)}-{x["DATE"].month:02d}-01", axis=1)
    pv_interp = [pv_data['CUMULATIVE_PV_INSTALLATIONS'].loc[k] for k in pv_key]

    rainfall_repaired = repair_dataframe(rainfall, "Rainfall amount (millimetres)" )
    rainfall_repaired.index = [f"{x.year}-{x.month:02d}-{x.day:02d}" for x in rainfall_repaired["DATE"]]
    rainfall_key = demand.apply(lambda x: f"{str(x["DATE"].year)}-{x["DATE"].month:02d}-{x["DATE"].day:02d}", axis=1)

    rainfall_interpolated = [rainfall_repaired["Rainfall amount (millimetres)"].loc[k] for k in rainfall_key]

    all_data = pd.DataFrame.from_dict({"datetime":demand["DATE"],
                                       "rainfall": rainfall_interpolated,
                                       "pv_capacity": pv_interp,
                                       "temperature": temperature_interpolated,
                                       "solar_power": solar_power_interpolated,
                                       "total_demand": demand["TOTALDEMAND"]})


    all_data.to_csv(cwd / "all_data_30min.csv", index=False)







