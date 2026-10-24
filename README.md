# Predictive Stock MLOps Project 🚀 (v3.3)

🌐 **Language / Мова:** [English] | [🇺🇦 Українська](README.uk.md)

---

An end-to-end MLOps project for automated financial data ingestion, daily model retraining (Continuous Training) optimized for each individual asset, artifact versioning in the Hugging Face Hub, and automated operational monitoring via Telegram.

In version **v3.3**, the system introduces an **AI Arbiter (`arbiter.py`)** that evaluates Random Forest vs. XGBoost head-to-head, computes a Win-Rate-weighted ensemble forecast, and delivers a single compact Telegram digest. It also upgrades the feature pipeline (`features.py`) to **33 fully stationary (relative) indicators** incorporating deep volatility triggers (`^VXN`, `^TNX`, `SOXX`, Garman-Klass volatility, Volatility Squeeze, and exact earnings proximity & EPS surprise radar) with a 20-day validation embargo buffer to eliminate look-ahead leakage.

---

## 🏗️ Architecture & ML Pipeline (Lifecycle)

The project implements a fully automated, fault-tolerant AI lifecycle divided into four independent infrastructure modules:

### 1. Data Ingestion Engine (`fetch_data.py`)
* Nightly fetches a **5-year rolling window** of historical market data via the Yahoo Finance API for a target list of equities.
* **Earnings Calendar & Surprise Radar:** Automatically collects historical and upcoming corporate earnings dates and EPS surprise metrics (`data/{ticker}_earnings.csv`).
* **Macro & Sector Volatility Context:** Concurrently ingests global market indicators: **S&P 500 (`^GSPC`)**, **CBOE VIX (`^VIX`)**, **Nasdaq-100 Volatility (`^VXN`)**, **US 10-Year Treasury Yield (`^TNX`)**, and the **Semiconductor ETF (`SOXX`)**.
* Automatically synchronizes and versions raw datasets (`.csv`) in the **Hugging Face Datasets** registry.

### 2. Continuous Training Engine (`train.py` and `train_xgb.py`)
* The training stage now runs in parallel A/B mode: two independent model circuits operate side by side for the same asset batch.
* **Contour A (`train.py`):** `RandomForestRegressor` with 200 decision trees and `max_depth=12`, optimized for a multi-output financial return forecast.
* **Contour B (`train_xgb.py`):** `XGBRegressor` wrapped in `MultiOutputRegressor` with regularization settings: `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`.
* **Fail-Safe Time Alignment:** Enforces explicit conversion of all indices to `UTC` and normalizes them to pure midnight using `.normalize()`, eliminating `NaN` anomalies during feature merging.
* **Deep Feature Engineering (20-Feature Matrix):** Dynamically constructs technical, calendar, macroeconomic, and fundamental data points.
* **Multi-Output Training:** Both models operate as multi-objective regressors. Training on relative percentage returns (`pct_change`), each model predicts a vector of three targets in a single forward pass: **1 day ahead (tomorrow)**, cumulative movement for **5 days ahead (trading week)**, and **20 business days ahead (monthly business trend)**.
* **Dual-Currency Validation:** Computes the model's Mean Absolute Error (MAE) for both horizons independently, converting percentage metrics into real USD value based on the asset's current price.
* **IPO Support:** Fresh equities with short histories are handled through elastic rolling windows using `min_periods=1` for `MA_200`, `MA_20`, `Volatility_5`, and `Volume_MA15`, allowing assets with as little as 40 sessions of history to remain trainable without collapsing into all-NaN feature tables.
* **Closed-Loop Integration:** At the start of each run, loads `evaluation_history.csv` from Hugging Face Datasets to incorporate retrospective error context into the next training cycle.
* **Dynamic Bias Correction:** Automatically computes the median error shift (Bias) over the past 14 days and compensates predictions with damping factor `0.2`, capped at ±0.5% for 1d and ±1.5% for 5d. If `VIX > 22`, the compensation is zeroed to avoid bias learned during turbulent panic regimes.
* **Confidence Badges & Win Rate:** Injects trust badges into the Telegram report based on the latest win-rate signal (🟢 WinRate ≥ 65%, 🟡 Moderate / limited data, 🔴 WinRate < 45%).
* Automatically pushes serialized model binaries into the **Hugging Face Model Registry** and dispatches a compact Markdown digest to **Telegram**.

### 3. Continuous Evaluation Engine (`evaluate.py` and `evaluate_xgb.py`)
* An autonomous system referee running nightly in a Docker container immediately after the training process.
* Both model tracks are evaluated in parallel: `evaluate.py` audits the Random Forest branch and `evaluate_xgb.py` audits the XGBoost branch.
* Streams the prediction logs (`predictions_history.csv` and `predictions_history_xgb.csv`) and matches them against benchmark ground truth closing prices harvested through the Yahoo Finance API.
* Computes Mean Absolute Error (MAE in USD and percentage metrics) along with Directional Accuracy (Win Rate) for 1d, 5d, and 20d forecasting horizons.
* Features absolute idempotency: tracks audited states via unique composite keys to prevent redundant verifications.
* Protects against non-trading days and exchange holidays by resolving the next actual market session instead of using a naive calendar BusinessDay shift.
* Automatically synchronizes the evaluation grid into `evaluation_history.csv` hosted in Hugging Face Datasets and fires an elastic, line-by-line chunked analytical feedback report to Telegram split at 3800 characters per message.

### 4. AI Arbiter & Ensemble Digest (`arbiter.py`)
* Compares Random Forest and XGBoost head-to-head across all historical evaluations (1d, 5d, 20d) and selects the reigning champion per equity over the latest 14-day window.
* Blends both models into a Win-Rate-weighted ensemble forecast.
* Replaces 4 verbose Telegram dumps with a single compact, scannable digest highlighting **Top Consensus Signals** (where both models agree 🤝 and WR ≥ 60%) followed by one-line summaries for remaining assets.

### 5. Public Client Inference (`predict.py`)
* A lightweight script for end-users or external integrations (on-demand inference).
* Operates token-free: streams the latest verified model weights (RF & XGBoost) from the Hugging Face Hub, builds the 24 stationary features (`features.py`), and outputs the ensemble forecast for **1d, 5d, and 20d**.

---

## 📊 Input Feature Matrix (33 Stationary Feature Columns in `features.py`)

The models use a strictly stationary, relative indicator stack free from raw price scale bias and look-ahead leakage:

| Category | Feature Name | Indicator Description |
| :--- | :--- | :--- |
| **Return Momentum** | `Daily_Return`, `Return_5d`, `Return_20d` | Relative percentage returns over 1, 5, and 20 sessions |
| **Trend Distance** | `Distance_to_MA5`, `Distance_to_MA20` | Relative price deviation from weekly and monthly SMAs |
| | `Distance_to_MA50`, `Distance_to_MA200` | Relative price deviation from quarterly and annual SMAs |
| **Deep Volatility** | `Volatility_5`, `Volatility_20` | 5-day and 20-day rolling return standard deviation |
| | `GK_Volatility` | Garman-Klass OHLC historical volatility estimator |
| | `Volatility_Squeeze` | Bollinger Band width vs Keltner ATR channel ratio (breakout spring detector) |
| **Session Structure** | `Intraday_Return`, `Day_Range`, `Gap` | Intra-session move, High/Low range, and overnight opening gap |
| **Calendar & Earnings** | `Day_of_Week`, `Earnings_Season` | Weekday index and corporate earnings season month flag |
| | `Earnings_Proximity` | Proximity radar to exact earnings date (Pre-earnings run-up / Post-earnings crush) |
| | `Last_EPS_Surprise` | Most recent reported quarterly EPS surprise percentage |
| **Volume & Alpha** | `Volume_Ratio` | Volume spike relative to 15-day average |
| | `SP500_Return`, `Relative_Return_1d` | Benchmark S&P 500 return and stock-specific daily alpha |
| | `SOXX_Return`, `Relative_to_SOXX` | Semiconductor sector return (`SOXX`) and stock alpha vs chipmakers |
| **Macro Fear Triggers** | `VIX_Close`, `VIX_Change` | Wall Street S&P 500 fear index level and daily change |
| | `VXN_Close`, `VXN_VIX_Spread` | Nasdaq-100 implied volatility (`^VXN`) and tech-fear spread over VIX |
| | `TNX_Change` | Daily yield momentum of US 10-Year Treasury bonds (`^TNX`) |
| **Oscillators & Shocks**| `RSI_14`, `MACD_Hist`, `BB_Position` | 14-day RSI, price-normalized MACD histogram, and Bollinger Band Z-score |
| | `ATR_Ratio`, `ATR_Pct` | True Range shock ratio (TR / ATR_14) and ATR as % of price |

---

## 🔧 Configuration & Environment Variables

The core pipeline is completely stateless and scales seamlessly without rebuilding the Docker image:

* `STOCK_TICKER` — Comma-separated list of target equities (e.g., `NVDA,GOOG,AAPL,MSFT,ASML,TSM`).
* `HF_TOKEN` — Hugging Face authentication token with `Write` access.
* `HF_REPO` — Target repository path for datasets (`username/predictive-stock-dataset`).
* `HF_MODEL_REPO` — Target repository path for model binaries (`username/predictive-stock-models`).
* `HF_MODEL_REPO_XGB` — Target repository path for XGBoost model binaries (`nadtoka/predictive-stock-models-xgb`).
* `TELEGRAM_BOT_TOKEN` — Bot authorization token issued by `@BotFather`.
* `TELEGRAM_CHAT_ID` — Your personal Telegram user account ID.

> 💡 **Fault Tolerance (Graceful Degradation):**
> * If executed without access tokens, the pipeline falls back to standalone local execution mode without crashing.
> * If the Yahoo Finance API encounters timeouts or errors while retrieving financial structures (`stock.info`), the system applies an engineering fallback, injecting default market neutral ratios to safeguard Cron execution.

---

## 🚀 Quick Start (Local Run)

### 1. Environment Setup
```bash
git clone https://github.com/nadtoka/predictive-stock-mlops.git
cd predictive-stock-mlops

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute the Pipeline Locally
```bash
export STOCK_TICKER="NVDA,ASML,AAPL"
export HF_TOKEN="your_hf_write_token"
export HF_REPO="nadtoka/predictive-stock-dataset"
export HF_MODEL_REPO="nadtoka/predictive-stock-models"
export TELEGRAM_BOT_TOKEN="your_tg_token"
export TELEGRAM_CHAT_ID="your_tg_id"

python fetch_data.py
python train.py
```

---

## 🐳 Running with Universal Docker Image

The immutable image behaves according to the execution arguments passed at runtime:

```bash
# Automated nightly data ingestion
docker run --rm \
  -e STOCK_TICKER="NVDA,ASML,AAPL" \
  -e HF_TOKEN="your_token" \
  -e HF_REPO="nadtoka/predictive-stock-dataset" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python fetch_data.py

# Automated nightly Multi-Output training & Telegram reporting
docker run --rm \
  -e STOCK_TICKER="NVDA,ASML,AAPL" \
  -e HF_TOKEN="your_token" \
  -e HF_MODEL_REPO="nadtoka/predictive-stock-models" \
  -e TELEGRAM_BOT_TOKEN="your_tg_token" \
  -e TELEGRAM_CHAT_ID="your_tg_id" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python train.py

# Automated nightly quality control & evaluation loop
docker run --rm \
  -e HF_TOKEN="your_token" \
  -e HF_REPO="nadtoka/predictive-stock-dataset" \
  -e TELEGRAM_BOT_TOKEN="your_tg_token" \
  -e TELEGRAM_CHAT_ID="your_tg_id" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python evaluate.py
```

---

## 📁 Project Structure

```text
predictive-stock-mlops/
├── .github/workflows/
│   └── docker-ci.yml       # Automated CI/CD (Build -> Smoke Test -> Push)
├── data/                   # Local raw historical cache (Git ignored)
├── models/                 # Local serialized model binaries (Git ignored)
├── models_xgb/             # Local cached XGBoost model binaries (Git ignored)
├── Dockerfile              # Instructions for building the immutable runtime
├── features.py             # Shared 33-stationary-feature pipeline, metrics & Telegram engine
├── fetch_data.py           # Data Ingestion module (Equities + Earnings + S&P 500 + VIX + VXN + TNX + SOXX)
├── run_pipeline.sh         # Main orchestration script (Fetch -> Evaluate -> Train -> Arbiter)
├── train.py                # Random Forest A-contour, 20-day validation embargo, HF sync
├── train_xgb.py            # XGBoost B-contour, multi-output training, HF sync
├── evaluate.py             # Automated Random Forest accuracy auditor from local data/ cache
├── evaluate_xgb.py         # Automated XGBoost accuracy auditor and market feedback loop
├── arbiter.py              # AI Arbiter (RF vs XGBoost vs Ensemble) & unified Telegram digest
├── predict.py              # Lightweight ensemble client inference (1d, 5d, and 20d forecasts)
└── requirements.txt        # Frozen dependency tree (scikit-learn, joblib, pandas, yfinance)
```

---

## Telegram Bot Interface

![Telegram Operational Report](https://github.com/user-attachments/assets/ff329bf4-1e32-4291-b358-413924483988)
