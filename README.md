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
3. **python**: contains pre- and post- processed datasets


## Prerequisites
All coding has been carried out in Python. Required modules are:
- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- seaborn
TODO: Craig - add additional


## Scripts
Descriptions of all Python scripts are as follows:
1. **data_wrangling.py**: inside root directory, used to extract datasets provided for the course from GitHub
2. **data_cleaning.py**: inside data folder, used to format and clean additional data
3. **SARIMAX.py**: 
4. **SARIMAX_30mins.py**: 
5. **SARIMAX_GBR_hybrid.py**: 
6. **aemo_forecast.py**: extract last AEMO peak electricity forecast for the following day
7. **colour_dict.py**: dictionary for consistent figure colours and axis labels
8. **data_import_export.py**: 
9. **data_manipulation.py**: 
10. **eda.py**: produce visualisations for exploratory data analysis
11. **eda_30min.py**: produce visualisations for exploratory data analysis
12. **final_test_lstm.py**: produce test predictions for final LSTM model
13. **final_test_lstm_analysis.py**: visualise test prediction performance for final LSTM model
14. **load_arima.py**: 
15. **lstm_best_models.py**: 
16. **lstm_final.py**: 
17. **lstm_final_shap.py**: 
18. **lstm functions.py**: refactored LSTM code used after first round of hyperparameter tuning
19. **lstm_hyperparam_tuning.py**: first round of LSTM hyperparameter tuning
20. **lstm_hyperparam_tuning_2.py**: subsequent LSTM model tuning
21. **lstm_initial_tests.py**: TODO: remove?
22. **public holidays.py**: 
23. **solar_irradiance_analysis.py**: 
24. **statistical_data_transformation.py**: 
25. **stats.py**: 

