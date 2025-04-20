# Stock Volatility Prediction Using Machine Learning

## Overview

This is an end-to-end machine learning project designed to identify high-volatility trading days across multiple publicly traded stocks. The model uses historical pricing data and engineered technical indicators to classify whether a given day is likely to experience a significant price swing.

The project is actively maintained and being expanded to support multiple tickers, additional features, and dashboard integration.

---

## Objectives

- Predict daily high-volatility events using historical price action
- Engineer meaningful features from time series financial data
- Compare baseline (Logistic Regression) and advanced models (Random Forest, XGBoost)
- Extend project to multiple stocks (TSLA, AMZN, NVDA, AAPL, etc.)
- Build interpretable and deployable machine learning tools for finance

---

## Core Features

- Real-time data ingestion using `yfinance`
- Volatility labeling based on intraday price swing thresholds
- Feature engineering:
  - Momentum and rolling averages
  - Daily return metrics
  - Volume trends and price range normalization
- Classification using:
  - Logistic Regression (baseline)
  - Random Forest (current best performer)
- Evaluation with:
  - Confusion matrices
  - Precision / recall / F1 scores
- Feature importance plots for interpretability

---

## Example Results (TSLA)

**Random Forest Classifier**  
- Accuracy: 98%  
- Volatility Recall: 89%  
- Precision: 100%  

The model demonstrated strong predictive power in identifying volatility spikes in TSLA over a one-year period.

---

## Repository Structure


---

## How to Use

```python
from src.run_model import run_volatility_model

run_volatility_model("TSLA")
run_volatility_model("NVDA")
run_volatility_model("AMZN")

Houston Molinaro
Data Science Student – University of Arkansas
Focused on predictive modeling, financial analytics, and ML-powered trading tools.
