# Import required libraries and custom functions
import pandas as pd
from pathlib import Path
import torch
import sys
import numpy as np
import public_holidays as ph
import lstm_functions as lf

#import lstm_functions2 as lf

# Set random seed for reproducibility
torch.manual_seed(0)

# Define file paths for data and results storage
cwd = Path.cwd()
root_folder = cwd.parent
data_folder = root_folder / 'data'


# Create output directory for model results
res_path = root_folder / 'Results' / 'LSTM' / 'Final' / 'Var Dropout'
res_path.mkdir(parents=True,exist_ok=True)


# Define modelling scenarios for different feature sets and datasets
scenarios = [
    {
        'name': 'all_variables_validation',
        'dropped_vars': 0,
        'test_data': False,
        'num_repeats_divisor': 48*20,
        'days_between_val': 20,
        'retrain': True
    },
    {
        'name': 'no_datetime_validation',
        'dropped_vars': 8,
        'test_data': False,
        'num_repeats_divisor': 48*20,
        'days_between_val': 20,
        'retrain': True
    },
    {
        'name': 'no_datetime_testing',
        'dropped_vars': 8,
        'test_data': True,
        'num_repeats_divisor': 48,
        'days_between_val': 1,
        'retrain': True
    }
]


# Loop through each modelling scenario
for scenario in scenarios:

    print('Running scenario:', scenario['name'])

    # Load dataset and define validation starting index
    df = pd.read_csv(data_folder / 'all_data_30min.csv')
    val_y_start_idx = 3408 + (48*2)

    # Preprocess data depending on validation or testing scenario
    if scenario['test_data']:
        df = lf.preprocess_30_min_data(df, True, val_data_only=False, test_data_only=True)
    else:
        df = lf.preprocess_30_min_data(df, True)

    # Prepare dataset and initialise results storage
    df, df_datetime, results = lf.eval_df(df)


    # Select feature set based on variable dropout configuration
    dropped_vars = scenario['dropped_vars']


    if dropped_vars == 0:
        #all vars:
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'year', 'hour', 'min', 'public_hol', 'month', 'day', 'month_sin',
            'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'min_sin',
            'min_cos']]
    elif dropped_vars == 1:
        #drop day values
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'year', 'hour', 'min', 'public_hol', 'month', 'month_sin',
            'month_cos', 'hour_sin', 'hour_cos', 'min_sin',
            'min_cos']]
    elif dropped_vars == 2:
    #drop month values
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'year', 'hour', 'min', 'public_hol', 'day', 
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos', 'min_sin',
            'min_cos']]
    elif dropped_vars == 3:
    # drop day and month
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'year', 'hour', 'min', 'public_hol',
            'hour_sin', 'hour_cos', 'min_sin',
            'min_cos']]
    elif dropped_vars == 4:
        # drop all date
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'hour', 'min', 'public_hol', 
            'hour_sin', 'hour_cos', 'min_sin',
            'min_cos']]
    elif dropped_vars == 5:
        # drop hour
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'year', 'min', 'public_hol', 'month', 'day', 'month_sin',
            'month_cos', 'day_sin', 'day_cos', 'min_sin',
            'min_cos']]
    elif dropped_vars == 6:
        # drop minute
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'year', 'hour', 'public_hol', 'month', 'day', 'month_sin',
            'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']]
    elif dropped_vars ==7:
        # drop all time
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand',
            'year', 'public_hol', 'month', 'day', 'month_sin',
            'month_cos', 'day_sin', 'day_cos']]
    elif dropped_vars == 8:
        # drop all datetime
        df = df[['rainfall', 'pv_capacity', 'temperature', 'solar_power', 'total_demand', 'public_hol']]



    # Calculate number of rolling windows based on scenario configuration
    num_repeats = round((len(df) - val_y_start_idx) / scenario['num_repeats_divisor'])

    # Train model using sliding window approach
    results = lf.repeat_windows(
        df,
        results,
        df_datetime,
        val_y_start_idx,
        num_repeats,
        scenario['days_between_val'],
        retrain=scenario['retrain']
    )

    # Save and display model performance metrics
    name = scenario['name'] + '.csv'
    lf.display_metrics(results, True, file_path=res_path, file_name=name)

    # Compute SHAP feature importance across training windows
    shap_results = lf.average_shap_training_windows(
        df,
        df_datetime,
        initial_val_y_start=val_y_start_idx,
        num_repeats=num_repeats,
        days_between_val=scenario['days_between_val'],
        retrain=scenario['retrain'],
        max_background=10,
        save=True,
        file_path=res_path,
        file_name_prefix=scenario['name']
    )

    # Output SHAP results
    print(shap_results)