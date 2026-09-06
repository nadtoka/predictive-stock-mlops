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
echo "🚀 STARTING MLOPS PIPELINE: $(date)"
echo "=============================================================================="

# Оновлення образу
echo "🔄 Перевірка та оновлення Docker образу..."
docker pull $IMAGE

# 🛰️ Крок 1: Збір даних (fetch_data.py)
echo "📡 1. Запуск збору даних для тікерів..."
docker run --rm \
  -e STOCK_TICKER="$STOCK_TICKER" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python fetch_data.py

if [ $? -ne 0 ]; then
    echo "❌ ПОМИЛКА: Збір даних завершився невдало. Пайплайн зупинено."
    exit 1
fi

# 🏗️ Крок 2: Тренування моделей та логування історії (train.py)
echo "🧠 2. Запуск тренування моделей та збереження історії прогнозів..."
docker run --rm \
  -e STOCK_TICKER="$STOCK_TICKER" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_MODEL_REPO="$HF_MODEL_REPO" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python train.py

if [ $? -ne 0 ]; then
    echo "❌ ПОМИЛКА: Тренування моделей завершилося невдало. Пайплайн зупинено."
    exit 1
fi

# 📊 Крок 3: Контроль якості та зворотний зв'язок (evaluate.py)
echo "📊 3. Запуск аудиту якості та розрахунку похибок моделей..."
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python evaluate.py

if [ $? -ne 0 ]; then
    echo "⚠️ ПОПЕРЕДЖЕННЯ: Аудит якості завершився з помилкою."
    exit 1
fi

echo "=============================================================================="
echo "✅ PIPELINE SUCCESSFULY FINISHED: $(date)"
echo "=============================================================================="