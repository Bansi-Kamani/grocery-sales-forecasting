# 🛒 Grocery Sales Forecasting — ABC Retail Company

> **Predicting multi-store, multi-product sales using an ensemble of 5 machine learning models across 54 stores and 33 product categories.**

---

## 📌 Project Overview

ABC Grocery, a South American retail chain, faced persistent challenges with inventory management due to inaccurate sales forecasts. This project builds a **fully automated, end-to-end ML pipeline** that predicts sales for every store–product combination using supervised learning — replacing manual, ad hoc forecasting methods.

**Target**: Predict daily sales for **July 31, 2017** and **August 15, 2017**  
**Dataset**: 4.5 years of sales data (Jan 2013 – Aug 2017) across 54 stores × 33 product types  
**Best Result**: ~80% of predictions achieved **MAE < 30 units**

---

## 🏗️ Architecture & Pipeline

```
Raw Data (Products_Information.csv)
        │
        ▼
┌─────────────────────┐
│   Data Preprocessing │  → datetime indexing, null checks, type casting
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Exploratory Data    │  → trend analysis, heatmaps, distribution plots
│  Analysis (EDA)      │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Segmentation        │  → 54 stores × 33 products = 1,782 unique segments
│  (Store × Product)   │
└─────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
Zero       Non-Zero
Sales      Sales
   │         │
   ▼         ▼
NumPy     5-Model Ensemble
zeros     (see below)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  → lag features (14/21/28 days), special offer lags
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Model Training &    │  → GridSearchCV, PredefinedSplit cross-validation
│  Hyperparameter      │
│  Optimization        │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Best Model          │  → min MAE selected per segment
│  Selection (per      │
│  segment)            │
└─────────────────────┘
        │
        ▼
     Results
```

---

## 🤖 Models Used

| Model | Key Strength | Lag Window |
|---|---|---|
| **XGBoost** | Gradient boosting with L1+L2 regularization | 28 days |
| **LightGBM** | GOSS + EFB for fast, memory-efficient training | 14 days |
| **Random Forest** | Bagging ensemble, robust to overfitting | 14 days |
| **MLP Regressor** | Captures non-linear relationships (4 hidden layers) | 21 days |
| **Linear Regression** | Baseline, fast, interpretable | 14 days |

Each model is trained **independently per store–product segment**, and the one with the **lowest MAE** is selected for final predictions.

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Segments with MAE < 30 | ~**80%** |
| Total unique segments | **1,782** |
| Data coverage | Jan 2013 – Aug 2017 |
| Evaluation metric | Mean Absolute Error (MAE) + Relative MAE |

> Relative MAE = `MAE / Average Sales` — used for fair comparison across high and low volume products.

---

## 🔧 Feature Engineering

- **Sales Lag Features**: Past 14/21/28 days of sales used as predictors
- **Promotional Lag Features**: Past offer strength values (same window)
- **Zero-sales filtering**: Initial zero-sale periods removed per segment before training
- **Temporal split**: 80% train / 20% validation, with test period Jul 31 – Aug 15, 2017

---

## 📁 Repository Structure

```
grocery-sales-forecasting/
│
├── forecasters.py                   # Core ML classes: SalesForecaster & ZeroSalesForecaster
├── grocery_sales_prediction.py      # Main pipeline: EDA, segmentation, prediction loop
├── Predicting_Sales_ABC_Grocery_Report.pdf  # Full academic report with methodology & results
├── Python_Slides_1207.pptx          # Project presentation slides
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
git clone https://github.com/Bansi-Kamani/grocery-sales-forecasting.git
cd grocery-sales-forecasting
pip install -r requirements.txt
```

### Run
```bash
# Place Products_Information.csv in the root directory, then:
python grocery_sales_prediction.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
```

Or simply:
```bash
pip install -r requirements.txt
```

---

## 🧠 Technical Highlights

- **OOP design**: `SalesForecaster` and `ZeroSalesForecaster` classes with clean APIs
- **Scalable**: Loops through all 1,782 store–product combinations automatically
- **Prevents data leakage**: Date set as index; strict train/validation/test temporal splits
- **Hyperparameter tuning**: GridSearchCV with PredefinedSplit (respects time ordering)
- **Handles edge cases**: Segments with all-zero sales handled separately via NumPy

---

## 📄 Report

A full academic report is included (`Predicting_Sales_ABC_Grocery_Report.pdf`) covering:
- Data description & preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Model selection rationale
- Performance evaluation & overfitting analysis
- Results with relative MAE comparisons

---

## 👤 Author

**Bansi Kamani**  
MSc Student — Mathematical Programming and Optimisation  
University of Manchester (2023–24)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/bansikamani/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Bansi-Kamani)

---

## 📜 License

This project is for academic and portfolio purposes.
