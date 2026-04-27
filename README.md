# Data Science Project: Group B
*To what extent does incorporating temperature, rainfall, solar irradiation, and PV uptake affect the performance of machine learning models for short-term predictions of daily peak electricity demand in NSW?*

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
1. **data_wrangling.py**: inside root directory, used to extract datasets provided for the course from GitHub
2. **data_cleaning.py**: inside data folder, used to format and clean additional data
3. **SARIMAX.py**: 
4. **SARIMAX_30mins.py**: 
5. **SARIMAX_GBR_hybrid.py**: 
6. **aemo_forecast.py**: extract last AEMO electricity forecasts from the previous day
7. **colour_dict.py**: dictionaries for consistent figure colours and axis labels
8. **data_import_export.py**: 
9. **data_manipulation.py**: 
10. **eda.py**: produce visualisations for exploratory data analysis
11. **eda_30min.py**: produce visualisations for exploratory data analysis
12. **ensemble_final.py**:
13. **final_test_lstm.py**: produce LSTM test predictions
14. **final_test_lstm_analysis.py**: visualise test prediction performance for models
15. **load_arima.py**: 
15. **lstm_best_models.py**: used in the final stage of initial hyperparameter tuning to compare input variable formats for current best models (Appendix Table C15)
18. **lstm functions.py**: refactored LSTM code providing functions used after initial hyperparameter tuning
18. **lstm functions_multi_seq.py**: used after testing, given the discovered overfitting, to investigate if passing multisequences to model reduced overfitting
18. **lstm functions_same_day_incl.py**: used in the final stage of initial hyperparameter tuning to compare the use of same-day input data for predictions (Appendix Table C15)
19. **lstm_hyperparam_tuning.py**: first round of LSTM hyperparameter tuning (Appendix Tables C6 through C13)
20. **lstm_hyperparam_tuning_2.py**: LSTM tuning after SHAP insights(Appendix Table C16)
21. **lstm_hyperparam_tuning_2-shap.py**:produce SHAP visualisation for three datasets: training set with temporal features, training set without temporal features and testing set without temporal features
22. **public_holidays.py**: 
23. **solar_irradiance_analysis.py**: 
24. **statistical_data_transformation.py**: 
25. **SARIMAX/final_plots.py**: Produces the final plots from the SARIMAX simulations
26. **SARIMAX/fitting.py** Runs the AutoARIMA procedure to determine the optimised parameters for the SARIMA/X models
27. **SARIMAX/fitting_local.py** Contains the runtime for ARIMA models potentially with a list of parameters to run for assessing models further
28. **SARIMAX/sliding_window.py**: Python code to produce the forecast data for the requested ARIMA model, can also produce the 30min intervals or just the daily maximuma
29. **SARIMA/window_testing.py**: code to test the effect of changing the window size. 

