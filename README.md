# Data Science Project: Group B
*To what extent does incorporating temperature, rainfall, solar irradiation, and PV instllations affect the performance of machine learning models for short-term predictions of daily peak electricity demand in NSW?*

### Members
- Jingwen Miao (z5630753)
- Molly Taylor (z5574769)
- Craig Booth (z5647543)
- Julien Sennane (z5393563)


## Repo Structure
1. **data**: contains pre- and post- processed datasets, and initial processing script for additional found data
2. **figures**: all figures produced for exploratory data analysis
3. **python**: contains scripts used in this project, alongside predictions inputted to and outputted from the ensemble
4. **Results**: contains testing visualisations for all mocels


## Prerequisites
All coding has been carried out in Python. Required modules are:
- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- seaborn
- statsforecast
- coreforecast
- scipy
- shap

### TODO: add ensemble folder - remove ensemble_final
## Scripts
Descriptions of all Python scripts are as follows:
1. **../data_wrangling.py**: inside root directory, used to extract datasets provided for the course from GitHub.
2. **../data/additional_data/data_cleaning.py**: inside data folder, used to format and clean additional data
3. **aemo_forecast.py**: extract last AEMO electricity forecasts from the previous day
4. **colour_dict.py**: dictionaries for consistent figure colours and axis labels
5. **data_import_export.py**: imports, repairs and interpolates data to the demand time intervals, and re-exports data for use in the LSTM and SARIMAX models as all_data_30min.csv
6. **data_manipulation.py**: imports, repairs and reduces the data to daily values exports values to all_data.csv
7. **eda.py**: produce visualisations for exploratory data analysis
8. **eda_30min.py**: produce visualisations for exploratory data analysis#
9. **final_test_lstm.py**: produce LSTM test predictions
10. **final_test_lstm_analysis.py**: visualise test prediction performance for models
11. **lstm_best_models.py**: used in the final stage of initial hyperparameter tuning to compare input variable formats for current best models (Appendix Table C15)
12. **lstm functions.py**: refactored LSTM code providing functions used after initial hyperparameter tuning
13. **lstm functions_multi_seq.py**: used after testing, given the discovered overfitting, to investigate if passing multisequences to model reduced overfitting
14. **lstm functions_same_day_incl.py**: used in the final stage of initial hyperparameter tuning to compare the use of same-day input data for predictions (Appendix Table C15)
15. **lstm_hyperparam_tuning.py**: first round of LSTM hyperparameter tuning (Appendix Tables C6 through C13)
16. **lstm_hyperparam_tuning_2.py**: LSTM tuning after SHAP insights(Appendix Table C16)
17. **lstm_hyperparam_tuning_2-shap.py**: produce SHAP visualisation for training and testings with different feature combinations
18. **SARIMAX/final_plots.py**: Produces the final plots from the SARIMAX simulations
19. **SARIMAX/fitting.py** Runs the AutoARIMA procedure to determine the optimised parameters for the SARIMA/X models
20. **SARIMAX/fitting_local.py** Contains the runtime for ARIMA models potentially with a list of parameters to run for assessing models further
22. **SARIMAX/sliding_window.py**: Python code to produce the forecast data for the requested ARIMA model, can also produce the 30min intervals or just the daily maximum
23. **SARIMA/window_testing.py**: code to test the effect of changing the window size. 
24. **Ensemble_files/Ensemble_model.py**: implementation of ensemble models
