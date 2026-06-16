# Predictive Stock MLOps Project 🚀

Енд-ту-енд (End-to-End) MLOps проєкт для автоматизованого збору фінансових даних, тренування прогнозних моделей та деплою сервісів передбачення вартості акцій.

---

## 🏗️ Поточна архітектура проєкту

Наразі реалізовано перший виробничий етап — **Automated Data Ingestion Pipeline**:
1. **Engine:** Модуль збору даних на Python (`yfinance` + `curl_cffi` для обходу TLS-блокувань Yahoo Finance). Підтримує мультитікерний збір даних (Multi-Ticker Ingestion).
2. **Containerization:** Скрипт повністю ізольований у легковаговому Docker-образі на базі `python:3.11-slim`.
3. **CI/CD & Automation:** Налаштовано GitHub Actions з використанням власного `self-hosted` раннера.
4. **Smoke Testing:** Пайплайн містить крок автоматичної валідації зібраних даних для кожного тікера перед пушем образу.
5. **Registry:** Готові імутабельні образи (тегуються за допомогою `short-SHA` комітів та `latest`) автоматично публікуються в **GitHub Container Registry (GHCR)**.

---

## 🔧 Конфігурація та параметризація

Пайплайн є абсолютно *stateless* та керується через змінні оточення:
* `STOCK_TICKER` — список тікерів через кому для одночасного збору даних (наприклад: `AAPL,MSFT,NVDA`).

У CI/CD списком цільових активів можна керувати через **GitHub Repository Variables** (`Settings -> Secrets and variables -> Actions` у вкладці `Variables`).

---

## 🚀 Швидкий старт

### 1. Локальний запуск через Docker (без встановлення Python)
Для запуску збору даних і збереження результатів на хост-машину використовуйте Docker Volumes:

```bash
mkdir -p data

docker run --rm \
  -e STOCK_TICKER="AAPL,MSFT,NVDA" \
  -v $(pwd)/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest
```

Після виконання контейнер самостійно видалиться, а в папці `data/` з'являться свіжі CSV-файли.

### 2. Локальне розгортання для розробки (З віртуальним оточенням)

```bash
# Клонування та перехід в репо
git clone [https://github.com/nadtoka/predictive-stock-mlops.git](https://github.com/nadtoka/predictive-stock-mlops.git)
cd predictive-stock-mlops

# Налаштування venv
python3 -m venv venv
source venv/bin/activate

# Встановлення залежностей
pip install -r requirements.txt

# Запуск збору даних
export STOCK_TICKER="AAPL,MSFT"
python fetch_data.py
```

---

## 📁 Структура проєкту

```text
predictive-stock-mlops/
├── .github/workflows/
│   └── docker-ci.yml       # СI/CD пайплайн (Build -> Smoke Test -> Push to GHCR)
├── data/                   # Локальний кеш даних (ігнорується Git)
├── Dockerfile              # Інструкція збірки абстрактного рушія даних
├── fetch_data.py           # Мультитікерний скрипт збору даних
├── requirements.txt        # Залежності (yfinance, pandas, curl_cffi)
└── README.md               # Документація проєкту
```
