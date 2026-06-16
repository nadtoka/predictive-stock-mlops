# Predictive Stock MLOps Project 🚀

Енд-ту-енд (End-to-End) MLOps проєкт для автоматизованого збору фінансових даних, щоденного перетренування прогнозних моделей (Continuous Training) під кожен актив окремо, версіонування артефактів у Hugging Face Hub та автоматичного моніторингу через Telegram.

---

## 🏗️ Архітектура та ML-Пайплайн (Lifecycle)

Проєкт реалізує повністю автоматизований, стійкий до помилок життєвий цикл ШІ, розділений на незалежні модулі:

1. **Data Ingestion Engine (`fetch_data.py`):**
   * Щоночі викачує актуальне **5-річне ковзне вікно (5-year Rolling Window)** історичних котирувань через Yahoo Finance API.
   * Повний перезапис вікна гарантує захист моделі від аномалій (наприклад, автоматично враховує спліти акцій типу корпоративного поділу Nvidia 1:10).
   * Автоматично синхронізує та версіонує сирі дані (`.csv`) у **Hugging Face Datasets** портфоліо.

2. **Continuous Training Engine (`train.py`):**
   * Запускається щоночі через системний Cron на ізольованій Runner-VM.
   * Прораховує технічні індикатори: ковзні середні (`MA_5`, `MA_20`), щоденне повернення (`Daily_Return`) та волатильність (`Volatility_5`).
   * **Ізоляція моделей:** Навчає *окремого спеціалізованого робота* `RandomForestRegressor` (100 дерев рішень) під кожен тікер індивідуально, враховуючи специфіку поведінки конкретної акції.
   * Зберігає ваги моделей локально в `.joblib` і автоматично пушить їх у публічний **Hugging Face Model Registry**.
   * Формує фінальний прогноз на наступний торговий день та надсилає красивий аналітичний Markdown-звіт у твій **Telegram**.

3. **Public Client Inference (`predict.py`):**
   * Легковажний скрипт для кінцевих користувачів або сторонніх сервісів.
   * Працює без жодних HF-токенів чи паролів: примусово стягує останні зафіксовані ваги моделей з Hugging Face Hub, робить швидкий запит до Yahoo Finance за останні 3 місяці (lookback період для прорахунку `MA_20`) і миттєво виводить прогноз на завтра.

---

## 🔧 Конфігурація та змінні оточення

Система є повністю *stateless* і гнучко масштабується без необхідності перезбірки Docker-образу. Список цільових компаній можна розширювати прямо в конфігах:

* `STOCK_TICKER` — список тікерів через кому (наприклад, великий Big Tech стек: `NVDA,GOOG,AAPL,MSFT,AMZN,ASML,ADBE,TSM,V,META`).
* `HF_TOKEN` — токен доступу до Hugging Face з правами `Write`.
* `HF_REPO` — шлях до репозиторію датасетів (наприклад: `username/predictive-stock-dataset`).
* `HF_MODEL_REPO` — шлях до репозиторію моделей (наприклад: `username/predictive-stock-models`).
* `TELEGRAM_BOT_TOKEN` — токен твого бота від `@BotFather`.
* `TELEGRAM_CHAT_ID` — твій особистий ID чату від `@userinfobot`.

> 💡 **Принцип плавного зниження (Graceful Degradation):** Якщо запуск відбувається без вказання токенів Hugging Face або Telegram (наприклад, при швидких локальних тестах), скрипти автоматично пропустять ці кроки, виконають роботу локально та не впадуть з помилкою.

---

## 🚀 Швидкий старт для розробки (Local Run)

### 1. Встановлення залежностей
```bash
git clone [https://github.com/nadtoka/predictive-stock-mlops.git](https://github.com/nadtoka/predictive-stock-mlops.git)
cd predictive-stock-mlops

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ручний запуск збору даних та тренування моделей для кількох тікерів
```bash
export STOCK_TICKER="NVDA,GOOG,AAPL"
export HF_TOKEN="your_hf_write_token"
export HF_REPO="nadtoka/predictive-stock-dataset"
export HF_MODEL_REPO="nadtoka/predictive-stock-models"
export TELEGRAM_BOT_TOKEN="your_tg_token"
export TELEGRAM_CHAT_ID="your_tg_id"

# Викачування 5 років історії для вказаних компаній
python fetch_data.py

# Навчання окремих моделей для кожної акції та пуш ваг на HF
python train.py
```

### 3. Публічний інференс (Перевірка прогнозів на завтра для будь-яких тікерів)
Будь-хто може використати мої натреновані моделі без авторизації. Достатньо просто передати список цільових компаній:

```bash
STOCK_TICKER="NVDA,GOOG,MSFT" python predict.py
```

---

## 🐳 Робота з універсальним Docker-образом

Зібраний у CI/CD образ є універсальним MLOps-комбайном. Залежно від переданої команди наприкінці, він виконує потрібну роль:

```bash
# Щонічний автоматичний збір даних (через Cron на сервері)
docker run --rm \
  -e STOCK_TICKER="NVDA,GOOG,AAPL,MSFT" \
  -e HF_TOKEN="your_token" \
  -e HF_REPO="nadtoka/predictive-stock-dataset" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python fetch_data.py

# Щонічне автоматичне тренування моделей та сповіщення в Telegram
docker run --rm \
  -e STOCK_TICKER="NVDA,GOOG,AAPL,MSFT" \
  -e HF_TOKEN="your_token" \
  -e HF_MODEL_REPO="nadtoka/predictive-stock-models" \
  -e TELEGRAM_BOT_TOKEN="your_tg_token" \
  -e TELEGRAM_CHAT_ID="your_tg_id" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python train.py
```

---

## 📁 Структура проєкту

```text
predictive-stock-mlops/
├── .github/workflows/
│   └── docker-ci.yml       # Автоматизований CI/CD пайплайн (Build без кешу -> Smoke Test -> Push)
├── data/                   # Локальний кеш 5-річної історії котирувань (ігнорується Git)
├── models/                 # Локальний кеш згенерованих ваг моделей .joblib (ігнорується Git)
├── Dockerfile              # Універсальна інструкція збірки імутабельного середовища
├── .dockerignore           # Виключення локальних venv та кешів зі збірки
├── .gitignore              # Захист бінарників моделей та даних від комітів у GitHub
├── fetch_data.py           # Модуль збору даних (5-річне ковзне вікно)
├── train.py                # Модуль Continuous Training, валідації та надсилання TG-звітів
├── predict.py              # Легковажний публічний клієнт для отримання прогнозів
└── requirements.txt        # Закріплені версії бібліотек (scikit-learn, joblib, pandas, yfinance)
```
