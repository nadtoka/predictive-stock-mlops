# Predictive Stock MLOps Project 🚀 (v3.0)

🌐 **Language / Мова:** [English] | [🇺🇦 Українська](README.uk.md)

---

An end-to-end MLOps project for automated financial data ingestion, daily model retraining (Continuous Training) optimized for each individual asset, artifact versioning in the Hugging Face Hub, and automated operational monitoring via Telegram.

In version **3.0**, the system has fully migrated to an autonomous evaluation loop (Feedback Loop), enabling not only training but also independent quality control of predictions throughout the production lifecycle.

---

## 🏗️ Architecture & ML Pipeline (Lifecycle)

The project implements a fully automated, fault-tolerant AI lifecycle divided into four independent infrastructure modules:

### 1. Data Ingestion Engine (`fetch_data.py`)
* Nightly fetches a **5-year rolling window** of historical market data via the Yahoo Finance API for a target list of equities.
* **Macro Context:** Concurrently ingests global market indicators: the benchmark US economic index **S&P 500 (`^GSPC`)** and the Wall Street fear index **CBOE Volatility Index (`^VIX`)**.
* Automatically synchronizes and versions raw datasets (`.csv`) in the **Hugging Face Datasets** registry.

### 2. Continuous Training Engine (`train.py`)
* Triggers nightly via system Cron on an isolated Runner-VM (Proxmox / Docker Swarm).
* **Fail-Safe Time Alignment:** Enforces explicit conversion of all indices to `UTC` and normalizes them to pure midnight using `.normalize()`, eliminating `NaN` anomalies during feature merging.
* **Deep Feature Engineering (16-Feature Matrix):** Dynamically constructs technical, calendar, macroeconomic, and fundamental data points.
* **Multi-Output Training:** The trained `RandomForestRegressor` (200 decision trees, `max_depth=12`) operates as a multi-objective regressor. Training on relative percentage returns (`pct_change`), it predicts a vector of two values in a single forward pass: market movement for **1 day ahead (tomorrow)** and cumulative movement for **5 days ahead (trading week)**.
* **Dual-Currency Validation:** Computes the model's Mean Absolute Error (MAE) for both horizons independently, converting percentage metrics into real USD value based on the asset's current price.
* Automatically pushes serialized model binaries into the **Hugging Face Model Registry** and dispatches a compact Markdown digest to **Telegram**.

### 3. Continuous Evaluation Engine (`evaluate.py`)
* An autonomous system referee running nightly in a Docker container immediately after the training process.
* Streams the prediction logs (`predictions_history.csv`) and matches them against benchmark ground truth closing prices harvested through the Yahoo Finance API.
* Computes Mean Absolute Error (MAE in USD and percentage metrics) along with Directional Accuracy (Win Rate) for both 1d and 5d forecasting horizons.
* Features absolute idempotency: tracks audited states via unique composite keys to prevent redundant verifications.
* Automatically synchronizes the evaluation grid into `evaluation_history.csv` hosted in Hugging Face Datasets and fires an elastic, line-by-line chunked analytical feedback report to Telegram.

### 4. Public Client Inference (`predict.py`)
* A lightweight script for end-users or external integrations (on-demand inference).
* Operates token-free: streams the latest verified model weights directly from the Hugging Face Hub, builds the corresponding feature graph locally for the current date using a 2-year lookback window (`period="2y"`), and instantly outputs the dual forecast to the console.

---

## 📊 Input Feature Matrix (16 Feature Columns)

The model utilizes a balanced blend of distinct feature categories:

| Category | Feature Name | Indicator Description |
| :--- | :--- | :--- |
| **Core Technicals** | `Close`, `Volume` | Current closing price and trading volume |
| | `MA_5`, `MA_20` | Short-term moving averages (weekly and monthly trends) |
| | `Daily_Return`, `Volatility_5` | Daily percentage return and historical volatility over 5 days |
| **Session Structure** | `Intraday_Return`, `Day_Range` | Intra-session price movement and trading amplitude (High/Low) |
| | `Gap` | Nightly price gap (opening price dislocation) |
| **Calendar-Based** | `Day_of_Week` | Day of the week index to capture "Friday profit-taking" patterns |
| **Volume Analytics**| `Volume_Ratio` | Volume spike tracking (current volume vs. 15-day average) |
| **Macro Context** | `SP500_Return`, `VIX_Close` | Benchmark S&P 500 returns and Wall Street implied volatility index |
| **Momentum** | `RSI_14` | Relative Strength Index to identify overbought/oversold regions |
| | `Distance_to_MA200` | Price deviation from the global long-term trend (200-day SMA) |
| **Fundamentals** | `Earnings_Season` | Corporate earnings season flag (active during Jan, Apr, Jul, Oct) |
| | `PE_Ratio`, `PS_Ratio` | Dynamic daily Price-to-Earnings and Price-to-Sales multipliers |
| | `Revenue_Growth` | Business scaling speed based on the latest quarterly reports |

---

## 🔧 Configuration & Environment Variables

The core pipeline is completely stateless and scales seamlessly without rebuilding the Docker image:

* `STOCK_TICKER` — Comma-separated list of target equities (e.g., `NVDA,GOOG,AAPL,MSFT,ASML,TSM`).
* `HF_TOKEN` — Hugging Face authentication token with `Write` access.
* `HF_REPO` — Target repository path for datasets (`username/predictive-stock-dataset`).
* `HF_MODEL_REPO` — Target repository path for model binaries (`username/predictive-stock-models`).
* `TELEGRAM_BOT_TOKEN` — Bot authorization token issued by `@BotFather`.
* `TELEGRAM_CHAT_ID` — Your personal Telegram user account ID.

> 💡 **Fault Tolerance (Graceful Degradation):**
> * If executed without access tokens, the pipeline falls back to standalone local execution mode without crashing.
> * If the Yahoo Finance API encounters timeouts or errors while retrieving financial structures (`stock.info`), the system applies an engineering fallback, injecting default market neutral ratios to safeguard Cron execution.

---

## 🚀 Quick Start (Local Run)

### 1. Environment Setup
```bash
git clone [https://github.com/nadtoka/predictive-stock-mlops.git](https://github.com/nadtoka/predictive-stock-mlops.git)
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
  -e HF_REPO="username/predictive-stock-dataset" \
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
├── Dockerfile              # Instructions for building the immutable runtime
├── fetch_data.py           # Data Ingestion module (Equities + S&P 500 + VIX)
├── train.py                # 16-feature assembly, Multi-Output training, dual MAE, TG engine
├── evaluate.py             # Automated model accuracy auditor, MAE % and Win Rate tracker
├── predict.py              # Lightweight client inference (On-demand 1d and 5d forecasts)
└── requirements.txt        # Frozen dependency tree (scikit-learn, joblib, pandas, yfinance)
```

---

## Telegram Bot Interface

![Telegram Operational Report](https://github.com/user-attachments/assets/ff329bf4-1e32-4291-b358-413924483988)
