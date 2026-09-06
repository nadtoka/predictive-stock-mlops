#!/bin/bash

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR="/opt/stock-mlops"
DATA_DIR="$BASE_DIR/data"
IMAGE="ghcr.io/nadtoka/predictive-stock-mlops:latest"

STOCK_TICKER="NVDA,GOOG,AAPL,MSFT,AMZN,ASML,ADBE,TSM,V,META,BULL,AMD,NET,QBTS,RGTI,IONQ,IBM,FIG,BILL,NFLX,AVGO,QQQM,UNH,FSLY,CAT,ETN,SNDK,SKHY,RKLB,INFQ,UBER,CHKP,MU,SNPS,CBRS"
HF_REPO="nadtoka/predictive-stock-dataset"
HF_MODEL_REPO="nadtoka/predictive-stock-models"
HF_MODEL_REPO_XGB="nadtoka/predictive-stock-models-xgb"

# ==============================================================================
# SECRETS CHECK
# ==============================================================================
if [ -f "$BASE_DIR/.env" ]; then
    source "$BASE_DIR/.env"
else
    echo "❌ ПОМИЛКА: Файл $BASE_DIR/.env не знайдено!"
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "=============================================================================="
echo "🚀 STARTING A/B MLOPS PIPELINE: $(date)"
echo "=============================================================================="

echo "🔄 Оновлення Docker образу..."
docker pull $IMAGE

# 🛰️ 1. Збір даних (Один раз для всіх)
echo "📡 1. Запуск збору даних..."
docker run --rm \
  -e STOCK_TICKER="$STOCK_TICKER" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_REPO="$HF_REPO" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python fetch_data.py

if [ $? -ne 0 ]; then
    echo "❌ ПОМИЛКА: Збір даних завершився невдало. Пайплайн зупинено."
    exit 1
fi

# 🌲 2. Контур A: RANDOM FOREST
echo "🧠 2A. Тренування Random Forest..."
docker run --rm \
  -e STOCK_TICKER="$STOCK_TICKER" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_MODEL_REPO="$HF_MODEL_REPO" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python train.py

echo "📊 2B. Аудит Random Forest..."
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python evaluate.py

# ⚡ 3. Контур B: XGBOOST
echo "⚡ 3A. Тренування XGBoost..."
docker run --rm \
  -e STOCK_TICKER="$STOCK_TICKER" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_MODEL_REPO_XGB="$HF_MODEL_REPO_XGB" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python train_xgb.py

echo "📊 3B. Аудит XGBoost..."
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python evaluate_xgb.py

echo "=============================================================================="
echo "✅ A/B PIPELINE FINISHED: $(date)"
echo "=============================================================================="