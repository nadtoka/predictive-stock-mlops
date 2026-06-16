# Predictive Stock MLOps Project 🚀 (v5.0)

Енд-ту-енд (End-to-End) MLOps проєкт для автоматизованого збору фінансових даних, щоденного перетренування прогнозних моделей (Continuous Training) під кожен актив окремо, версіонування артефактів у Hugging Face Hub та автоматичного моніторингу через Telegram. 

У версії 5.0 модель отримала повноцінний макроекономічний зір та інструменти технічного аналізу моментуму ринку.

---

## 🏗️ Архітектура та ML-Пайплайн (Lifecycle)

Проєкт реалізує повністю автоматизований, стійкий до помилок життєвий цикл ШІ, розділений на незалежні модулі:

1. **Data Ingestion Engine (`fetch_data.py`):**
   * Щоночі викачує актуальне **5-річне ковзне вікно (5-year Rolling Window)** історичних котирувань через Yahoo Finance API для цільового списку акцій.
   * **Макро-контекст:** Паралельно викачує глобальні ринкові індикатори: головний індекс американської економіки **S&P 500 (`^GSPC`)** та офіційний індекс страху й паніки Волл-стріт **CBOE Volatility Index (`^VIX`)**.
   * Повний перезапис вікна гарантує захист моделі від аномалій та корпоративних сплітів.
   * Автоматично синхронізує та версіонує сирі дані (`.csv`) у **Hugging Face Datasets** портфоліо.

2. **Continuous Training Engine (`train.py`):**
   * Запускається щоночі через системний Cron на ізольованій Runner-VM (Proxmox).
   * **Захист від часових зсувів (Fail-Safe Time Alignment):** Примусово конвертує індекси всіх джерел у єдиний формат `UTC` та нормалізує їх за допомогою `.normalize()` до чистої календарної півночі, запобігаючи утворенню `NaN` при злитті таблиць.
   * **Глибокий інженерний Feature Engineering (14 ознак):**
     * *Технічні базові:* Ковзні середні за тиждень та місяць (`MA_5`, `MA_20`), добова доходність (`Daily_Return`), волатильність/нервозність за 5 днів (`Volatility_5`).
     * *Внутрішньоденна структура:* Рух ціни всередині сесії (`Intraday_Return`), амплітуда торгів (`Day_Range`), нічний стрибок ціни (`Gap`).
     * *Поведені та календарні:* День тижня (`Day_of_Week`) для відловлювання «п'ятничних фіксацій», смарт-об'єм торгів (`Volume_Ratio`) для трекінгу вливань великих фондів відносно 15-денного середнього.
     * *Макроекономіка та Моментум:* Ринкова доходність (`SP500_Return`), рівень світової паніки (`VIX_Close`) та індекс відносної сили для детекції перекупленості/перепроданості (`RSI_14`).
   * **Ізоляція та конфігурація моделей:** Навчає *окремого спеціалізованого робота* `RandomForestRegressor` подвоєної потужності (**200 дерев рішень**, `max_depth=12`, `min_samples_leaf=3` для жорсткої боротьби з ринковим шумом) під кожен тікер індивідуально.
   * Зберігає ваги моделей у `.joblib` і автоматично пушить їх у **Hugging Face Model Registry**.
   * Формує фінальний прогноз на наступну торгову сесію та надсилає детальний аналітичний Markdown-звіт із зазначенням чесної похибки (MAE) у твій **Telegram**.

3. **Public Client Inference (`predict.py`):**
   * Легковажний скрипт для кінцевих користувачів або сторонніх сервісів.
   * Працює без жодних HF-токенів чи паролів: примусово стягує останні зафіксовані ваги моделей з Hugging Face Hub, локально прораховує аналогічний математичний граф фіч і миттєво виводить прогноз на завтра.

---

## 🔧 Конфігурація та змінні оточення

Система є повністю *stateless* і гнучко масштабується без необхідності перезбірки Docker-образу. Список цільових компаній можна розширювати прямо в конфігах:

* `STOCK_TICKER` — список тікерів через кому (наприклад, великий Big Tech стек: `NVDA,GOOG,AAPL,MSFT,AMZN,ASML,ADBE,TSM,V,META`).
* `HF_TOKEN` — токен доступу до Hugging Face з правами `Write`.
* `HF_REPO` — шлях до репозиторію датасетів (наприклад: `username/predictive-stock-dataset`).
* `HF_MODEL_REPO` — шлях до репозиторію моделей (наприклад: `username/predictive-stock-models`).
* `TELEGRAM_BOT_TOKEN` — токен твого бота від `@BotFather`.
* `TELEGRAM_CHAT_ID` — твій особистий ID чату від `@userinfobot`.

> 💡 **Принцип плавного зниження (Graceful Degradation / Відмовостійкість):**
> * Якщо запуск відбувається без вказання токенів Hugging Face або Telegram, скрипти виконають роботу локально та не впадуть з помилкою.
> * Якщо під час тренування файл макро-показників (наприклад, VIX) буде пошкоджено або не знайдено, система автоматично застосує нейтральний інженерний fallback (`VIX_Close = 15.0`), зберігши стабільність розрахунку.

---

## 🚀 Швидкий старт для розробки (Local Run)

### 1. Встановлення залежностей
```bash
git clone https://github.com/nadtoka/predictive-stock-mlops.git
cd predictive-stock-mlops

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ручний запуск збору даних та тренування моделей
```bash
export STOCK_TICKER="NVDA,GOOG,AAPL"
export HF_TOKEN="your_hf_write_token"
export HF_REPO="nadtoka/predictive-stock-dataset"
export HF_MODEL_REPO="nadtoka/predictive-stock-models"
export TELEGRAM_BOT_TOKEN="your_tg_token"
export TELEGRAM_CHAT_ID="your_tg_id"

# Викачування 5 років історії акцій + S&P500 + VIX
python fetch_data.py

# Масштабне тренування моделей (200 дерев, 14 фіч, валідація MAE за останні 20 днів)
python train.py
```

---

## 🐳 Робота з універсальним Docker-образом

Зібраний у CI/CD образ є універсальним MLOps-комбайном. Залежно від переданої команди наприкінці, він виконує потрібну роль:

```bash
# Щонічний автоматичний збір даних (включаючи макро-індекси)
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
├── data/                   # Кеш історії котирувань та індексів (^GSPC, ^VIX) (ігнорується Git)
├── models/                 # Локальний кеш ваг моделей .joblib (ігнорується Git)
├── Dockerfile              # Універсальна інструкція збірки імутабельного середовища
├── .dockerignore           # Виключення локальних venv та кешів зі збірки
├── .gitignore              # Захист бінарників моделей та даних від комітів у GitHub
├── fetch_data.py           # Модуль збору даних (Акції + S&P 500 + VIX)
├── train.py                # Модуль Feature Engineering (14 фіч, RSI), навчання лісу та TG-репортингу
├── predict.py              # Легковажний публічний клієнт для отримання безпарольних прогнозів
└── requirements.txt        # Фіксовані версії бібліотек (scikit-learn, joblib, pandas, yfinance, curl_cffi)
```

---

## Інтерфейс Telegram бота

![Telegram Operational Report](https://github.com/user-attachments/assets/2bba71be-3227-4fdb-92d3-0d999b37a3e8)

