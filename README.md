# Ankit_paradox_aiPredictive Paradox — Electricity Demand Forecasting

A machine learning pipeline to forecast the next hour's electricity demand (demand_mw) on Bangladesh's national grid

Problem Overview
The goal is to predict demand_mw at time t+1 using only information available up to time t. The model must use classical ML only — no deep learning, no ARIMA, no Prophet.
Three datasets were provided:

PGCB_date_power_demand.xlsx — hourly demand and generation data (target source)
weather_data.xlsx — hourly temperature, humidity, and precipitation
economic_full_1.csv — annual macroeconomic indicators from World Bank


Pipeline Summary
1. Data Preparation & Anomaly Handling
The raw PGCB data had several real-world issues that needed to be resolved before any modeling:
Duplicates: The dataset contained multiple entries for the same timestamp. These were resolved by taking the mean of all duplicate rows for each timestamp.
Half-hourly entries: Some records were logged at 30-minute intervals instead of hourly. These were aggregated to hourly frequency using mean values.

Anomaly detection: The demand series had severe undocumented spikes. A rolling IQR-based strategy was used — any value falling outside 3× the interquartile range of a local rolling window was flagged as an outlier and replaced via linear interpolation from neighboring valid values.

Missing values: After reindexing to a clean hourly frequency, gaps were filled using forward-fill followed by backward-fill, ensuring no future data was used to impute past values (zero leakage).

Economic data integration: The annual macroeconomic data (GDP, population, access to electricity, etc.) was merged into the hourly series by matching each row's calendar year to the corresponding annual value — a simple and logical way to bring long-term economic context into an hourly feature set.

3. Feature Engineering
Since tree-based models treat each row independently, the concept of "time" had to be built entirely through engineered features.
Calendar features: Hour of day, day of week, month, quarter, year trend, weekend flag, season, is_peak_hour (18–21h), is_morning_ramp (6–9h).
Lag features: Past demand values at 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, 168h (1 week), and 336h (2 weeks) lags. Same-hour lags from 7 and 14 days ago were also included to capture weekly seasonality, and neighbour lagging is included to make it more accurate.

Rolling aggregates: Rolling means and standard deviations over 3h, 6h, 12h, 24h, and 168h windows — all computed on shifted data to prevent leakage.
Demand trend features: 1-hour and 24-hour demand differences, 24h and 7-day momentum, 6-hour linear slope (demand_slope_6h), and demand concavity (rate of change of slope).

Peer-hour statistics: For each hour of the day, mean and standard deviation of demand over the past 7 days of the same hour, plus a z-score of the current value relative to that peer group.
Daily load profile clustering: K-Means clustering on the previous day's 24-hour demand profile, used to assign a "load shape type" label to each row — giving the model context about what kind of day yesterday was.

Weather features: Temperature, humidity, and precipitation joined from weather_data.xlsx on the hourly timestamp.
All features were computed strictly using past data only. No future information was used at any point.

5. Train / Test Split
Data was sorted chronologically. The full year 2023 was held out as the test set. Everything before 2024 was used for training and hyperparameter tuning. This ensures strict chronological separation with zero data leakage.

6. Models Used
Two classical gradient boosting models were trained and compared:

XGBoost — trained with n_estimators=1000, learning_rate=0.05, and tree regularization parameters.
LightGBM — trained with base parameters and also tuned using Optuna over 50 trials (LightGBM_Tuned).

The best model was selected based on MAPE on the chronological test set.

5. Evaluation
Primary metric: Mean Absolute Percentage Error (MAPE) on the 2024 hold-out test set.
Additional diagnostics:

Residual distribution (MW error histogram)
Percentage error distribution
MAPE broken down by hour of day to identify when the model struggles most


6. Feature Importance
The best model's feature_importances_ were used to identify the key drivers of grid demand. Recent lag features (especially lag_1h, lag_24h, lag_168h) and rolling averages consistently ranked among the top predictors, confirming that recent historical demand is the strongest signal for short-term forecasting.

How to Run

Install dependencies:

bash   pip install pandas numpy scikit-learn xgboost lightgbm optuna openpyxl matplotlib seaborn

Place the three dataset files in the same directory as the notebook.
Run all cells in order from top to bottom.
