# Stock Volatility Prediction Using Machine Learning

## Overview

This is an end-to-end machine learning project designed to identify high-volatility trading days across multiple publicly traded stocks. The model uses historical pricing data and engineered technical indicators to classify whether a given day is likely to experience a significant price swing.

The project is actively maintained and being expanded to support multiple tickers, additional features, and dashboard integration.

---

## Objectives

- Predict daily high-volatility events using historical price action
- Engineer meaningful features from time series financial data
- Compare baseline (logistic regression) and advanced models (Random Forest, XGBoost)
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
- Evaluation with confusion matrices, precision, recall, and F1 scores
- Feature importance plots for interpretability

---

## Results (Tesla Model)

**Random Forest Classifier**
- Accuracy: 98%
- Volatility Recall: 89%
- Precision: 100%

The model demonstrated strong predictive power in identifying volatility spikes in TSLA over a one-year period.

---

## Roadmap

### Multi-Ticker Support
- [x] Tesla (TSLA)
- [ ] Amazon (AMZN)
- [ ] NVIDIA (NVDA)
- [ ] Apple (AAPL)
- [ ] Microsoft (MSFT)

### Feature Expansion
- [ ] Technical indicators from `pandas-ta` (e.g., RSI, MACD, Bollinger Bands)
- [ ] Macroeconomic inputs (CPI, Fed statements)
- [ ] News and sentiment signals (Reddit, Twitter, headlines)

### Usability & Deployment
- [ ] Modular pipeline for multi-stock training
- [ ] Dashboard integration (Streamlit or Gradio)
- [ ] Exportable daily prediction summaries
- [ ] Continuous data updates via scheduling

---

## Tech Stack

- Python (Pandas, NumPy, Matplotlib)
- scikit-learn (Logistic Regression, Random Forest)
- yFinance (live stock data)
- Streamlit / Plotly (planned)
- pandas-ta, NLP (planned)

---

## Repository Structure (Planned)
