# Ankit_paradox_ai
paradox_notebook.ipynb

Overview

This repository contains the complete machine learning pipeline for the Predictive Paradox project. The pipeline is designed to build a highly robust predictive model using advanced feature engineering, comprehensive data cleaning, and gradient boosting frameworks (XGBoost/CatBoost). A major focus of this project is model interpretability and understanding the underlying drivers of the target variable.

Interpretation & Reporting
1. Handling Missing Data and Anomalies
Time-Series Continuity (Missing Data): Missing numerical values were primarily addressed using forward-fill (ffill) interpolation to preserve temporal continuity without leaking future information. For non-sequential categorical or static numerical gaps, median imputation was utilized to prevent skewness from outliers.

Anomaly Detection & Capping: Extreme outliers and sudden spikes in the data were identified using the Interquartile Range (IQR) method and rolling z-scores with threshold 4. To prevent data loss while maintaining model stability, anomalous values were capped at the statistical upper and lower bounds rather than being entirely removed.

2. Engineered Features (Temporal & External)
To maximize the predictive power of the model, several new dimensions were added to the dataset:

Time-Based Features: Extracted basic temporal indicators such as 'Day of Week', 'Month', and 'Is_Weekend'. Why: To capture inherent seasonal cycles and weekly operational trends within the data.

Lagged Features: Engineered multiple lag intervals (e.g., 7-day, 14-day lags) for the primary variables. Why: Time-series predictions rely heavily on historical context; lags allow the model to natively understand what happened in the immediate past.

Rolling Statistics: Calculated rolling means and rolling volatility (standard deviation) over moving windows.To capture short-term momentum and measure the stability/instability of the market or system at any given point as the future values mainly depend on present fluctuations .

External Features: Integrated relevant external macroeconomic or domain-specific indicators (e.g., variance indices). Why: To provide the model with an external context that drives sudden shifts in the target variable which historical lags alone cannot explain.

3. Feature Importance
Using the model's native feature importance methods (and SHAP values for validation), the following key drivers were identified:

Dominance of Lagged Variables: Recent historical lags emerged as the most critical predictors, indicating a strong autocorrelation in the dataset. The immediate past is the heaviest driver of near-future outcomes.

Impact of Rolling Volatility: Engineered rolling standard deviation features ranked significantly high in importance. This proves that the rate of change and recent instability are major factors in driving the model's decisions during fluctuating periods.

External Feature Contribution: External indices provided a crucial, albeit secondary, contribution. Their importance peaked during anomalous periods, helping the model correct its trajectory when internal historical trends were disrupted.
