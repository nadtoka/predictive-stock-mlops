#!/bin/bash

# Визначаємо шлях до директорії проєкту
BASE_DIR="/opt/stock-mlops"
DATA_DIR="$BASE_DIR/data"
IMAGE="ghcr.io/nadtoka/predictive-stock-mlops:latest"

# Завантажуємо секрети з локального .env файлу, якщо він існує
if [ -f "$BASE_DIR/.env" ]; then
    source "$BASE_DIR/.env"
else
    echo "❌ ПОМИЛКА: Файл $BASE_DIR/.env не знайдено!"
    exit 1
fi

# ==============================================================================
# CONFIGURATION
# ==============================================================================
STOCK_TICKER="NVDA,GOOG,AAPL,MSFT,AMZN,ASML,ADBE,TSM,V,META,BULL,AMD,NET,QBTS,RGTI,IONQ,IBM,FIG,DJT,BILL,NFLX,AVGO,QQQM,UNH,FSLY,CAT,ETN"
HF_REPO="nadtoka/predictive-stock-dataset"
HF_MODEL_REPO="nadtoka/predictive-stock-models"

DATA_DIR="/opt/stock-mlops/data"
IMAGE="ghcr.io/nadtoka/predictive-stock-mlops:latest"

echo "=============================================================================="
echo "🚀 STARTING MLOPS PIPELINE: $(date)"
echo "=============================================================================="

# Пулл останнього образу про всяк випадок
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

# Перевіряємо, чи успішно відпрацював перший крок
if [ $? -ne 0 ]; then
    echo "❌ ПОМИЛКА: Збір даних завершився невдало. Тренування скасовано."
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

# 📊 Крок 3: Контроль якості та зворотний зв'язок (evaluate.py)
echo "📊 3. Запуск аудиту якості та розрахунку похибок моделей..."
docker run --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_REPO="$HF_REPO" \
  -e TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
  -e TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
  -v "$DATA_DIR":/app/data \
  $IMAGE python evaluate.py

echo "=============================================================================="
echo "✅ PIPELINE SUCCESSFULY FINISHED: $(date)"
echo "=============================================================================="