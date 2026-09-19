# 🛍️ Customer Churn Prediction & Retention Engine

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application designed to evaluate subscription and retail customer churn risk in real-time, featuring robust modular data pipelines and an interactive Streamlit dashboard.

</div>

---

## 📌 Project Overview
Customer churn represents one of the largest revenue leaks for subscription-based businesses. This project goes beyond a standard exploratory notebook by delivering a **production-ready machine learning system**, featuring:
1. **Modular Data Pipelines:** Clean separation of data loading, feature engineering, and inference preprocessing using Scikit-Learn `Pipeline` and `ColumnTransformer`.
2. **Balanced Classification:** Optimized Random Forest model tuned with class-weight balancing to maximize precision and recall on high-risk accounts.
3. **Interactive Web Interface:** A sleek, real-time Streamlit dashboard allowing stakeholders to simulate customer metrics and instantly view churn probabilities and actionable retention strategies.

---

## 🚀 Project Architecture
```text
churn-prediction-engine/
│
├── data/
│   └── telco_churn.csv         # Raw customer dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning and custom feature engineering transformers
│   └── train.py                # Model training script, evaluation metrics, and joblib serialization
├── app.py                      # Interactive Streamlit frontend web app
├── model_pipeline.pkl          # Serialized production-ready Scikit-Learn pipeline
└── requirements.txt            # Project Python dependencies