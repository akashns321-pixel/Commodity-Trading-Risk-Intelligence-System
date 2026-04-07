================================================================================
  COMMODITY TRADING RISK PREDICTION — README
  Project: Commodity Trading Risk Intelligence System
  File:    risk_prediction.py
================================================================================

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------
This script is a full end-to-end machine learning pipeline that analyzes
historical commodity futures price data (Crude Oil, Brent Crude, Heating Oil,
Natural Gas, RBOB Gasoline) spanning 2000–2024.

It performs:
  - Exploratory Data Analysis (EDA) and data validation
  - Feature engineering (rolling volatility, momentum, price shocks, etc.)
  - Binary risk classification (High Risk vs. Normal)
  - Model training: Logistic Regression + Random Forest (tuned)
  - SHAP-based explainability
  - A template-based hedging recommendation engine
  - Model serialization for deployment

--------------------------------------------------------------------------------
PREREQUISITES
--------------------------------------------------------------------------------
Python version required: Python 3.8 or higher

Install all required libraries by running:

    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels shap joblib

Full list of dependencies used in the script:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - statsmodels
  - shap
  - joblib
  - warnings (built-in)

--------------------------------------------------------------------------------
INPUT DATA
--------------------------------------------------------------------------------
The script expects a CSV file with the following details:

  File name : all_fuels_data.csv
  Columns   : date, ticker, commodity, open, high, low, close, volume

Update the file path on line 26 of risk_prediction.py to point to your
local CSV file before running:

    df = pd.read_csv(r"C:\Users\akash\Desktop\wise\all fuel\all_fuels_data.csv")

Replace the path above with your actual file location. For example:

    Windows : df = pd.read_csv(r"C:\Users\YourName\data\all_fuels_data.csv")
    macOS   : df = pd.read_csv("/Users/YourName/data/all_fuels_data.csv")
    Linux   : df = pd.read_csv("/home/YourName/data/all_fuels_data.csv")

--------------------------------------------------------------------------------
HOW TO EXECUTE
--------------------------------------------------------------------------------

OPTION 1 — Run as a Python script (command line)
-------------------------------------------------
Step 1: Open a terminal or command prompt.

Step 2: Navigate to the folder containing risk_prediction.py:
    cd path/to/your/project/folder

Step 3: Run the script:
    python risk_prediction.py

Note: All plots will appear one at a time. Close each plot window to allow
the script to continue to the next step.


OPTION 2 — Run in Jupyter Notebook (recommended for interactive use)
---------------------------------------------------------------------
Step 1: Convert the script to a notebook (only needed once):
    pip install nbformat
    jupyter nbconvert --to notebook --execute risk_prediction.py

    OR open Jupyter and manually paste the code cell by cell.

Step 2: Launch Jupyter Notebook:
    jupyter notebook

Step 3: Open the notebook file and run cells using Shift + Enter,
        or click "Run All" from the Cell menu.


OPTION 3 — Run in VS Code
--------------------------
Step 1: Open risk_prediction.py in Visual Studio Code.
Step 2: Install the Python extension if not already installed.
Step 3: Right-click inside the file and select "Run Python File in Terminal",
        OR click the Run button (▶) at the top right of the editor.

--------------------------------------------------------------------------------
EXECUTION ORDER
--------------------------------------------------------------------------------
The script is designed to run sequentially from top to bottom. Do NOT skip
sections, as each phase depends on outputs from the previous one.

  Phase 1  — Library imports and data loading
  Phase 2  — Date parsing, sorting, gap analysis
  Phase 3  — Data quality validation (nulls, duplicates, negative prices)
  Phase 4  — Price history plots and moving average overlays
  Phase 5  — Returns and rolling volatility calculation
  Phase 6  — Univariate and bivariate analysis
  Phase 7  — Cross-commodity correlation and clustering
  Phase 8  — Time-series decomposition (trend, seasonality, residual)
  Phase 9  — Target variable definition (forward 10-day volatility)
  Phase 10 — Feature engineering (momentum, price shock, demand anomaly, etc.)
  Phase 11 — VIF check and feature selection
  Phase 12 — Train/test chronological split (80/20, shuffle=False)
  Phase 13 — IsolationForest anomaly detection (fitted on train only)
  Phase 14 — Logistic Regression baseline model
  Phase 15 — Random Forest with TimeSeriesSplit GridSearchCV tuning
  Phase 16 — Model evaluation (AUC, classification report, confusion matrix)
  Phase 17 — SHAP explainability (global, waterfall, per-commodity)
  Phase 18 — Hedging recommendation engine
  Phase 19 — Model serialization (joblib)

--------------------------------------------------------------------------------
OUTPUTS
--------------------------------------------------------------------------------
After a successful run, the following files will be saved to the working
directory (the folder where you ran the script):

  commodity_risk_model.pkl  — Trained Random Forest model (deployment-ready)
  feature_scaler.pkl        — StandardScaler fitted on training data

To load these artifacts later for inference without retraining:

    import joblib
    best_rf = joblib.load('commodity_risk_model.pkl')
    scaler  = joblib.load('feature_scaler.pkl')

--------------------------------------------------------------------------------
EXPECTED RESULTS
--------------------------------------------------------------------------------
  Best Model       : Random Forest (tuned via TimeSeriesSplit GridSearch)
  Best Parameters  : max_depth=6, n_estimators=300, min_samples_leaf=2
  Cross-Val AUC    : 0.8015 (5-fold TimeSeriesSplit)
  Test AUC         : 0.849
  High-Risk Recall : 0.782
  F1 Score         : 0.726
  Top SHAP Drivers : volatility_30d, volatility_7d, momentum_7

--------------------------------------------------------------------------------
IMPORTANT NOTES
--------------------------------------------------------------------------------
1. Negative Crude Oil prices (April 2020) are genuine market events — the
   historic WTI collapse. They are retained intentionally. Simple pct_change()
   returns are used instead of log returns for this reason.

2. The train/test split is strictly chronological (shuffle=False) to prevent
   data leakage. Do not change this.

3. The risk threshold (75th percentile) and IsolationForest are fitted on
   training data only to avoid leakage from the test period.

4. Natural Gas shows a ~54% high-risk label rate due to the global pooled
   threshold. Per-commodity thresholds are recommended for production use.

5. The git commands at the end of the script (git init, git push, etc.) are
   optional and only needed if you want to push the project to GitHub.
   These can be safely ignored or removed.

--------------------------------------------------------------------------------
CONTACT / REPOSITORY
--------------------------------------------------------------------------------
GitHub:https://github.com/akashns321-pixel/Commodity-Trading-Risk-Intelligence-System.git

================================================================================
