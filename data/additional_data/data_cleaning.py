
import pandas as pd


# NOTE: existing data ranges from 01/01/2010 to 18/03/2021 in NSW

def format_date(df):
    """
    replace Year, Month, (Day) columns with DATE in current range
    returns edited dataframe
    """
    if 'Day' in df.columns: # bankstown data
        df['DATE'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df = df[(df['DATE'] >= pd.Timestamp('2010-01-01')) & (df['DATE'] <= pd.Timestamp('2021-03-18'))].reset_index(drop=True)
        df = df.drop(columns=['Year', 'Month', 'Day'])
    else: # pv data
        df['DATE'] = pd.to_datetime(df['Month'], format='%Y-%m')
        df = df[(df['DATE'] >= pd.Timestamp('2010-01-01')) & (df['DATE'] <= pd.Timestamp('2021-03-18'))].reset_index(drop=True)
        df = df.drop(columns='Month')
    return df


## DAILY RAINFALL - BANKSTOWN ##
rain_df = pd.read_csv('original_data\daily_rainfall_bankstown_data.csv')
print(rain_df.head())
rain_df = rain_df[['Year', 'Month',
       'Day', 'Rainfall amount (millimetres)']]
rain_df = rain_df.rename(columns={'Rainfall amount (millimetres)': 'DAILY_RAINFALL'})
rain_df = format_date(rain_df)

# check missing values
print(rain_df.isna().any())
# small proportion of missing measurements
print('Missing rainfall measurements: ', rain_df['DAILY_RAINFALL'].isna().sum())
print('Present rainfall measurements: ', rain_df['DAILY_RAINFALL'].notna().sum())
# interpolate missing values 
rain_df['DAILY_RAINFALL'] = rain_df['DAILY_RAINFALL'].interpolate(method='linear')
# NOTE: should consider different approaches, just using linear for efficiency

rain_df.to_csv('processed_data\daily_rainfall_bankstown.csv')


## DAILY RAINFALL - BANKSTOWN ##
solar_df = pd.read_csv('original_data\daily_solar_exposure_bankstown_data.csv')
print(solar_df.head())
solar_df = solar_df[['Year', 'Month',
       'Day', 'Daily global solar exposure (MJ/m*m)']]
solar_df = solar_df.rename(columns={'Daily global solar exposure (MJ/m*m)': 'DAILY_SOLAR_EXPOSURE'})
solar_df = format_date(solar_df)

# check missing values 
print(solar_df.isna().any())
print('Missing solar measurements: ', solar_df['DAILY_SOLAR_EXPOSURE'].isna().sum())
print('Present solar measurements: ', solar_df['DAILY_SOLAR_EXPOSURE'].notna().sum())
# interpolate missing values 
solar_df['DAILY_SOLAR_EXPOSURE'] = solar_df['DAILY_SOLAR_EXPOSURE'].interpolate(method='linear')

solar_df.to_csv('processed_data\daily_solar_exposure_bankstown.csv')


## PV INSTALLATION COUNT - NSW  ##
pv_count_df = pd.read_csv('original_data\monthly_pv_installations_nsw_data.csv')
print(pv_count_df.head())
pv_count_df = pv_count_df.rename(columns={'Installations': 'PV_INSTALLATIONS'})
pv_count_df = format_date(pv_count_df)

print(pv_count_df.isna().any())


# NOTE: this cumulative measurement may be unreliable depending on PV uninstallation rates
pv_count_df['CUMULATIVE_PV_INSTALLATIONS'] = pv_count_df['PV_INSTALLATIONS'].cumsum()
pv_count_df.to_csv('processed_data\monthly_pv_installations_nsw.csv')


## PV INSTALLATION CAPACITY - NSW ##
pv_capacity_df = pd.read_csv('original_data\monthly_pv_installation_capacity_nsw_data.csv')
print(pv_capacity_df.head())
pv_capacity_df = pv_capacity_df.rename(columns={'Capacity (kW)': 'PV_CAPACITY'})
pv_capacity_df = format_date(pv_capacity_df)

print(pv_capacity_df.isna().any())

# NOTE: this cumulative measurement may be unreliable depending on PV uninstallation rates
pv_capacity_df['CUMULATIVE_PV_CAPACITY'] = pv_capacity_df['PV_CAPACITY'].cumsum()
pv_capacity_df.to_csv('processed_data\monthly_pv_capacity_nsw.csv')

