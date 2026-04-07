# Commodity Trading Risk Intelligence System

## 📌 Overview

This project analyzes historical commodity futures price data (2000–2024) to identify and predict risk patterns using machine learning and statistical techniques.

It implements a complete pipeline from data preprocessing and analysis to model training and explainability.

---

## 🚀 Key Features

* Exploratory Data Analysis (EDA)
* Data validation and preprocessing
* Feature engineering (volatility, momentum, price shocks)
* Binary risk classification (High Risk vs Normal)
* Machine Learning Models:

  * Logistic Regression (baseline)
  * Random Forest (tuned)
* SHAP-based explainability
* Hedging recommendation system
* Model saving for reuse (joblib)

---

## 📊 Dataset

* File: `all_fuels_data.csv`
* Columns: date, ticker, commodity, open, high, low, close, volume
* Commodities:

  * Crude Oil
  * Brent Oil
  * Natural Gas
  * Heating Oil
  * RBOB Gasoline

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Statsmodels
* SHAP
* Joblib

---

## 📁 Project Structure

* `commodity_risk_prediction.py` → Main script
* `Commodity_risk_prediction.ipynb` → Notebook with outputs
* `all_fuels_data.csv` → Dataset
* `Commodity_Risk_Prediction_Project_Report.pdf` → Report

---

## ▶️ How to Run

### Run using Python

```bash
python commodity_risk_prediction.py
```

### Run in VS Code

* Open `.py` file
* Click **Run ▶️**

### Run in Jupyter

* Open `.ipynb`
* Click **Run All**

---

## 📈 Results

* Best Model: Random Forest (tuned)
* Test AUC: ~0.85
* Effective high-risk detection
* Key drivers identified using SHAP

---

## ⚠️ Notes

* Chronological train-test split (no data leakage)
* Negative oil prices (2020) are retained as real market behavior
* Script must be run sequentially

---

## 👤 Author

Akash N S
