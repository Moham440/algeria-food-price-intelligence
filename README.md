# 🌾 Algeria Food Price Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end Data Science platform for monitoring, detecting anomalies in,
and forecasting food prices across Algerian regions.**

*Data sources: FAO · WFP · HDX*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dashboard Screenshots](#️-dashboard-screenshots)
- [ML Models](#-ml-models)
- [Results Interpretation](#-results-interpretation)
- [Configuration](#-configuration)
- [Docker Deployment](#-docker-deployment)
- [Running Tests](#-running-tests)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🌍 Overview

The **Algeria Food Price Intelligence System** is a production-grade data science
platform that:

1. **Collects** real food price data from the FAO FAOSTAT API and WFP/HDX datasets.
2. **Cleans & Engineers** features including Ramadan seasonality, rolling statistics,
   and regional price ratios.
3. **Detects Anomalies** using an Isolation Forest + One-Class SVM ensemble to flag
   sudden price spikes.
4. **Forecasts Prices** using a Prophet + LSTM ensemble model with up to 24-month
   horizons.
5. **Visualises** everything in an interactive Streamlit dashboard with choropleth
   maps, trend charts, and an alert board.

### Monitored Products

| Category | Products |
|---|---|
| 🥦 Vegetables | Tomates · Pommes de terre · Oignons · Carottes |
| 🍗 Proteins | Poulet · Viande rouge (mouton & bœuf) · Oeufs |
| 🌾 Cereals | Pain · Semoule · Farine · Riz |
| 🫙 Staples | Huile végétale · Sucre · Lait · Beurre |

### Covered Regions (Wilayas)

Alger · Oran · Constantine · Annaba · Tlemcen · Sétif · Blida · Batna · Biskra · Béjaïa

---

## 🏗️ Architecture

```
algeria-food-price-intelligence/
├── data/
│   ├── raw/          # Original FAO/WFP Parquet files
│   ├── processed/    # Cleaned + feature-engineered data
│   └── external/     # Weather, CPI, exchange rate data
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_price_forecasting.ipynb
├── src/
│   ├── data_ingestion/
│   │   ├── fao_connector.py   # FAOSTAT API client
│   │   └── wfp_connector.py   # WFP/HDX data downloader
│   ├── preprocessing/
│   │   ├── data_cleaner.py    # Missing values, outliers, deduplication
│   │   └── feature_engineering.py  # Temporal, lag, rolling features
│   ├── models/
│   │   ├── anomaly_detector.py  # IsolationForest + OneClassSVM
│   │   └── price_predictor.py   # Prophet + LSTM + Ensemble
│   └── utils/
│       └── helpers.py           # Shared utilities
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── tests/
│   ├── test_data_cleaner.py
│   ├── test_anomaly_detector.py
│   └── test_helpers.py
├── config/
│   └── config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ✨ Features

### 1. Data Collection
- ✅ FAO FAOSTAT REST API connector with retry + rate-limiting
- ✅ WFP/HDX CSV downloader with graceful fallback
- ✅ Disk-cache decorator to avoid redundant API calls
- ✅ Parquet storage for fast I/O

### 2. Preprocessing
- ✅ Missing value imputation (interpolation / mean / median / drop)
- ✅ IQR, Z-score, and Isolation Forest outlier removal
- ✅ Ramadan calendar integration (price demand spikes)
- ✅ Algerian public holiday detection
- ✅ Monthly aggregation by product × wilaya
- ✅ Cyclical encoding of month and day-of-week

### 3. Anomaly Detection
- ✅ **Isolation Forest** (unsupervised, tree-based)
- ✅ **One-Class SVM** (kernel-based boundary)
- ✅ Composite anomaly score [0, 1]
- ✅ Severity tiers: LOW / MEDIUM / HIGH / CRITICAL
- ✅ CSV/Excel alert export

### 4. Price Forecasting
- ✅ **Prophet** with Algerian holidays and Ramadan seasonality
- ✅ **LSTM** (2-layer stacked with dropout, TensorFlow/Keras)
- ✅ **Ensemble** weighted average of both models
- ✅ Confidence intervals (90% by default)
- ✅ Evaluation metrics: MAE · RMSE · MAPE

### 5. Dashboard
- ✅ 🗺️ Interactive bubble map (Plotly Mapbox) — prices by wilaya
- ✅ 📈 Multi-product time-series chart with Ramadan shading
- ✅ 🌡️ Price heatmap (region × month)
- ✅ 🚨 Colour-coded anomaly alert table
- ✅ 🔮 On-demand price prediction widget
- ✅ 📥 One-click CSV / Excel export

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip 23+
- (Optional) Docker 24+

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourorg/algeria-food-price-intelligence.git
cd algeria-food-price-intelligence

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Set environment variables
cp .env.example .env
# Edit .env to add API keys if needed

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501**.

---

## 💻 Usage

### Data Ingestion

```python
from src.data_ingestion.fao_connector import FAOConnector
from src.data_ingestion.wfp_connector import WFPConnector

# Fetch and save FAO price data
fao = FAOConnector()
fao_path = fao.fetch_and_save(year_start=2015, year_end=2024)
print(f"FAO data saved: {fao_path}")

# Fetch WFP market data
wfp = WFPConnector()
wfp_path = wfp.fetch_and_save()
print(f"WFP data saved: {wfp_path}")
```

### Preprocessing Pipeline

```python
import pandas as pd
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer

raw = pd.read_parquet("data/raw/wfp/wfp_algeria_prices.parquet")

cleaner = DataCleaner()
clean = cleaner.clean(raw)

engineer = FeatureEngineer()
features = engineer.build_features(clean)
print(features.columns.tolist())
```

### Anomaly Detection

```python
from src.models.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
detector.fit(features)
scored = detector.predict(features)

alerts = detector.generate_alerts(scored)
for a in alerts[:5]:
    print(a.message)

# Save model
detector.save("anomaly_v1")
```

### Price Forecasting

```python
from src.models.price_predictor import EnsemblePredictor

predictor = EnsemblePredictor()
predictor.fit(features, product="Tomates", region="Alger")

forecast = predictor.predict(features, product="Tomates", region="Alger", n_steps=12)
print(forecast[["ds", "ensemble_yhat", "yhat_lower", "yhat_upper"]])

# Evaluate
metrics = predictor.evaluate(features, product="Tomates", region="Alger")
for model, m in metrics.items():
    print(f"{model}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, MAPE={m['MAPE']:.1f}%")
```

---

## 🖥️ Dashboard Screenshots

| Tab | Description |
|-----|-------------|
| **🗺️ Carte** | Bubble map showing latest price intensity per wilaya |
| **📈 Tendances** | Multi-product time-series with Ramadan shading + heatmap |
| **🚨 Alertes** | Colour-coded anomaly alert table with export |
| **🔮 Prédictions** | Interactive forecast with confidence interval |
| **📥 Export** | Raw data table + CSV/Excel download |

---

## 🤖 ML Models

### Anomaly Detection

| Algorithm | Hyperparameter | Default |
|---|---|---|
| Isolation Forest | `contamination` | 0.05 |
| Isolation Forest | `n_estimators` | 100 |
| One-Class SVM | `nu` | 0.05 |
| One-Class SVM | `kernel` | rbf |

The composite score is the mean of the two normalised decision functions.
Rows with score ≥ 0.70 are flagged as anomalies.

### Price Forecasting

| Model | Key Features |
|---|---|
| Prophet | Algerian holidays, Ramadan regressor, multiplicative seasonality |
| LSTM | 2-layer stacked (128→64 units), dropout=0.2, look-back=60 months |
| Ensemble | 50% Prophet + 50% LSTM (configurable) |

---

## 📊 Results Interpretation

### Anomaly Scores

| Score Range | Severity | Meaning |
|---|---|---|
| 0.95 – 1.00 | 🔴 CRITICAL | Extreme spike; likely supply shock or data error |
| 0.85 – 0.95 | 🟠 HIGH | Significant abnormal price increase |
| 0.75 – 0.85 | 🟡 MEDIUM | Moderate deviation; monitor closely |
| 0.70 – 0.75 | 🟢 LOW | Mild anomaly; likely seasonal or market fluctuation |

### Forecast Confidence

The 90% confidence interval widens as the forecast horizon increases.
Intervals that grow rapidly indicate high price volatility for that
product/region combination.

---

## ⚙️ Configuration

All settings live in `config/config.yaml`.  Key sections:

```yaml
preprocessing:
  missing_value_strategy: "interpolate"  # interpolate | mean | median | drop
  outlier_method: "iqr"                  # iqr | zscore | isolation_forest
  outlier_threshold: 3.0

anomaly_detection:
  alert_threshold: 0.7   # lower = more sensitive

forecasting:
  prophet:
    forecast_periods: 30
  ensemble:
    prophet_weight: 0.5
    lstm_weight: 0.5
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t algeria-food-price:latest .

# Run dashboard
docker run -p 8501:8501 algeria-food-price:latest

# With persistent data volume
docker run -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    algeria-food-price:latest
```

### Docker Compose (with Redis cache)

```yaml
# docker-compose.yml
version: "3.9"
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_data_cleaner.py -v

# Fast (exclude slow integration tests)
pytest tests/ -v -m "not slow"
```

---

## 📦 Project Structure (full)

```
algeria-food-price-intelligence/
├── config/
│   └── config.yaml              # All project settings
├── dashboard/
│   └── app.py                   # Streamlit multi-tab dashboard
├── data/
│   ├── raw/fao/                 # FAO Parquet files
│   ├── raw/wfp/                 # WFP Parquet files
│   ├── processed/               # Cleaned + features
│   └── external/                # CPI, weather, exchange rates
├── models/saved/
│   ├── anomaly/                 # Saved anomaly detector
│   └── forecasting/             # Saved Prophet + LSTM models
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_price_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── fao_connector.py
│   │   └── wfp_connector.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_cleaner.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py
│   │   └── price_predictor.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaner.py
│   ├── test_anomaly_detector.py
│   └── test_helpers.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit your changes following [Conventional Commits](https://conventionalcommits.org)
4. Open a Pull Request

Please ensure all tests pass and code is PEP8-compliant (`flake8 src/`).

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [FAO FAOSTAT](https://www.fao.org/faostat/) for agricultural statistics
- [WFP VAM](https://vam.wfp.org/) for market price monitoring
- [HDX](https://data.humdata.org/) for open humanitarian data
- [Prophet](https://facebook.github.io/prophet/) by Meta Research
- [Streamlit](https://streamlit.io/) for the dashboard framework

---

<div align="center">
Made with Chalabi Mohammed El Amine for food security intelligence in Algeria
</div>
