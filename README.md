# Predictive Stock MLOps Project 🚀 (v2.0)

Енд-ту-енд (End-to-End) MLOps проєкт для автоматизованого збору фінансових даних, щоденного перетренування прогнозних моделей (Continuous Training) під кожен актив окремо, версіонування артефактів у Hugging Face Hub та автоматичного моніторингу через Telegram. 

У версії 2.0 система повністю перейшла на патерн Multi-Output Regression, дозволяючи однією моделлю одночасно прогнозувати два часових горизонти, а також отримала гібридний зір, що об'єднує технічний та корпоративний фундаментальний аналіз.

---

## 🏗️ Архітектура та ML-Пайплайн (Lifecycle)

Проєкт реалізує повністю автоматизований, стійкий до помилок життєвий цикл ШІ, розділений на три незалежні інфраструктурні модулі:

1. Data Ingestion Engine (fetch_data.py):
   * Щоночі викачує актуальне 5-річне ковзне вікно (5-year Rolling Window) історичних котирувань через Yahoo Finance API для цільового списку акцій.
   * Макро-контекст: Паралельно викачує глобальні ринкові індикатори: головний індекс американської економіки S&P 500 (^GSPC) та індекс страху Волл-стріт CBOE Volatility Index (^VIX).
   * Автоматично синхронізує та версіонує сирі дані (.csv) у Hugging Face Datasets портфоліо.

2. Continuous Training Engine (train.py):
   * Запускається щоночі через системний Cron на ізольованій Runner-VM (Proxmox / Docker Swarm).
   * Fail-Safe Time Alignment: Примусово конвертує індекси всіх джерел у єдиний формат UTC та нормалізує їх через .normalize(), запобігаючи утворенню NaN при злитті таблиць.
   * Глибокий Feature Engineering (Матриця з 16 ознак): Система динамічно збирає технічні, календарні, макроекономічні та фундаментальні фактори.
   * Multi-Output Навчання: Навчений RandomForestRegressor (200 дерев рішень, max_depth=12) працює в режимі багатоцільової регресії. Модель вчиться на відсоткових доходностях (pct_change) і за один прохід прогнозує вектор із двох значень: рух на 1 день наперед (завтра) та кумулятивний рух на 5 днів наперед (робочий тиждень).
   * Двовалютна валідація: Розраховує похибку моделі (MAE) для обох горизонтів окремо, переводячи відсотки помилки в реальні долари (USD) від поточної ціни активу.
   * Автоматично пушить готові бінарники у Hugging Face Model Registry та надсилає компактний Markdown-звіт у Telegram.

3. Public Client Inference (predict.py):
   * Легковажний скрипт для кінцевих користувачів або сторонніх сервісів (on-demand інференс).
   * Працює без токенів: стягує останні зафіксовані ваги моделей з Hugging Face Hub, локально прораховує аналогічний математичний граф фіч для поточної дати за останні 2 роки історії (period="2y") і миттєво виводить подвійний прогноз у консоль.

---

## 📊 Матриця вхідних ознак (16 Feature Columns)

Для ухвалення рішень модель використовує збалансований стек ознак:

| Категорія | Назва фічі | Опис індикатора |
| :--- | :--- | :--- |
| Технічні базові | Close, Volume | Поточна ціна закриття та об'єм торгів |
|  | MA_5, MA_20 | Короткострокові ковзні середні (тиждень та місяць) |
|  | Daily_Return, Volatility_5 | Добова доходність та рівень нервозності ринку за 5 днів |
| Структура сесії | Intraday_Return, Day_Range | Рух ціни всередині сесії та амплітуда (High/Low) торгів |
|  | Gap | Розмір нічного стрибка ціни (розрив відкриття) |
| Календарні | Day_of_Week | День тижня для врахування «п'ятничних фіксацій» прибутку |
| Об'єми торгів | Volume_Ratio | Сплекс торгів (поточний об'єм відносно 15-денного середнього) |
| Макро-контекст | SP500_Return, VIX_Close | Доходність індексу S&P 500 та рівень паніки Волл-стріт |
| Моментум | RSI_14 | Індекс відносної сили для детекції зон перегріву |
|  | Distance_to_MA200 | Відстань ціни від глобального річного тренду (200-денна середня) |
| Фундаментал | Earnings_Season | Прапорець сезону квартальних звітів корпорацій (1, 4, 7, 10 місяці) |
|  | PE_Ratio, PS_Ratio | Динамічні щоденні коефіцієнти Price-to-Earnings та Price-to-Sales |
|  | Revenue_Growth | Швидкість масштабування бізнесу за останніми даними компанії |

---

## 🔧 Конфігурація та змінні оточення

Система є повністю stateless і гнучко масштабується без необхідності перезбірки Docker-образу:

* STOCK_TICKER — список тікерів через кому (наприклад: NVDA,GOOG,AAPL,MSFT,ASML,TSM).
* HF_TOKEN — токен доступу до Hugging Face з правами Write.
* HF_REPO — репозиторій датасетів (username/predictive-stock-dataset).
* HF_MODEL_REPO — репозиторій моделей (username/predictive-stock-models).
* TELEGRAM_BOT_TOKEN — токен твого бота від @BotFather.
* TELEGRAM_CHAT_ID — твій особистий ID чату.

> 💡 Відмовостійкість (Graceful Degradation):
> * Якщо запуск відбувається без вказання токенів, скрипти виконають роботу локально та не впадуть.
> * Якщо API Yahoo Finance поверне помилку або таймаут під час запиту фінансового інфо (stock.info), система застосує безпечний інженерний fallback, виставивши дефолтні коефіцієнти, що збереже працездатність Крону.

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

### 2. Локальний запуск пайплайну
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

## 🐳 Робота з універсальним Docker-образом

Зібраний у CI/CD образ виконує потрібну роль залежно від переданої команди:

```bash
# Щонічний автоматичний збір даних
docker run --rm \
  -e STOCK_TICKER="NVDA,ASML,AAPL" \
  -e HF_TOKEN="your_token" \
  -e HF_REPO="nadtoka/predictive-stock-dataset" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python fetch_data.py

# Щонічне Multi-Output тренування та надсилання звіту в Telegram
docker run --rm \
  -e STOCK_TICKER="NVDA,ASML,AAPL" \
  -e HF_TOKEN="your_token" \
  -e HF_MODEL_REPO="nadtoka/predictive-stock-models" \
  -e TELEGRAM_BOT_TOKEN="your_tg_token" \
  -e TELEGRAM_CHAT_ID="your_tg_id" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python train.py
```

## 📁 Структура проєкту

```
predictive-stock-mlops/
├── .github/workflows/
│   └── docker-ci.yml       # CI/CD пайплайн (Build без кешу -> Smoke Test -> Push)
├── data/                   # Кеш історії котирувань та індексів (ігнорується Git)
├── models/                 # Кеш ваг моделей .joblib (ігнорується Git)
├── Dockerfile              # Універсальна інструкція збірки імутабельного середовища
├── fetch_data.py           # Модуль збору даних (Акції + S&P 500 + VIX)
├── train.py                # Клієнтський Multi-Output тренер на 16 фіч
├── predict.py              # Публічний інференс на 2 роки історії (прогноз на 1д та 5д)
└── requirements.txt        # Фіксовані версії бібліотек
```
---

## Інтерфейс Telegram бота

![Telegram Operational Report](https://github.com/user-attachments/assets/294eab36-8eba-4c4f-a4ac-627086d5a06c)
