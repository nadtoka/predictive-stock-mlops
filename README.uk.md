# Predictive Stock MLOps Project 🚀 (v3.2)

🌐 **Мова / Language:** [🇬🇧 English](README.md) | [Українська]

Енд-ту-енд (End-to-End) MLOps проєкт для автоматизованого збору фінансових даних, щоденного перетренування прогнозних моделей (Continuous Training) під кожен актив окремо, версіонування артефактів у Hugging Face Hub та автоматичного моніторингу через Telegram. 

У версії **v3.2** система розгортає паралельний A/B-турнір моделей Random Forest і XGBoost, а також додає нативну підтримку свіжих IPO-активів через еластичні ковзні вікна і стабільну обробку короткої історії без падіння через NaN.

---

## 🏗️ Архітектура та ML-Пайплайн (Lifecycle)

Проєкт реалізує повністю автоматизований, стійкий до помилок життєвий цикл ШІ, розділений на чотири незалежні інфраструктурні модулі:

1. Data Ingestion Engine (fetch_data.py):
   * Щоночі викачує актуальне 5-річне ковзне вікно (5-year Rolling Window) історичних котирувань через Yahoo Finance API для цільового списку акцій.
   * Макро-контекст: Паралельно викачує глобальні ринкові індикатори: головний індекс американської економіки S&P 500 (^GSPC) та індекс страху Волл-стріт CBOE Volatility Index (^VIX).
   * Автоматично синхронізує та версіонує сирі дані (.csv) у Hugging Face Datasets портфоліо.

2. Continuous Training Engine (train.py та train_xgb.py):
   * Тренувальний контур тепер працює в паралельному A/B-режимі: два незалежні канали моделей обробляють один і той самий пакет активів одночасно.
   * Контур A (train.py): RandomForestRegressor з 200 деревами рішень, оптимізований для мультивихідного прогнозу повернень.
   * Контур B (train_xgb.py): XGBRegressor, обгорнутий у MultiOutputRegressor з регуляризацією: max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8.
   * Fail-Safe Time Alignment: Примусово конвертує індекси всіх джерел у єдиний формат UTC та нормалізує їх через .normalize(), запобігаючи утворенню NaN при злитті таблиць.
   * Глибокий Feature Engineering (Матриця з 16 ознак): Система динамічно збирає технічні, календарні, макроекономічні та фундаментальні фактори.
   * Multi-Output Навчання: Обидві моделі працюють у режимі багатоцільової регресії. Модель вчиться на відсоткових доходностях (pct_change) і за один прохід прогнозує вектор із двох значень: рух на 1 день наперед (завтра) та кумулятивний рух на 5 днів наперед (робочий тиждень).
   * Двовалютна валідація: Розраховує похибку моделі (MAE) для обох горизонтів окремо, переводячи відсотки помилки в реальні долари (USD) від поточної ціни активу.
   * Підтримка свіжих IPO: молоді активи з короткою історією обробляються через еластичні ковзні середні з min_periods=1 для MA_200, MA_20, Volatility_5 та Volume_MA15, що дозволяє навчати активи з історією від 40 сесій без падіння через NaN.
   * **Closed-Loop Integration:** На початку роботи завантажує `evaluation_history.csv` з Hugging Face Datasets, щоб врахувати ретроспективні помилки у наступному циклі навчання.
   * **Dynamic Bias Correction:** Автоматично розраховує медіанний зсув похибки (Bias) за останні 14 днів і компенсує прогнози з демпфінгом 0.2, капом ±0.5% для 1d та ±1.5% для 5d. Якщо VIX > 22, коригування скидається в нуль, щоб уникнути навчання на помилках в періоди паніки.
   * **Confidence Badges & Win Rate:** Додає до Telegram-звіту бейджі надійності на основі актуального Win Rate (🟢 WinRate ≥ 65%, 🟡 Moderate / мало даних, 🔴 WinRate < 45%).
   * Автоматично пушить готові бінарники у Hugging Face Model Registry та надсилає компактний Markdown-звіт у Telegram.

3. Continuous Evaluation Engine (evaluate.py та evaluate_xgb.py):
   * Автономний рефері системи, який запускається щоночі в Docker-контейнері слідом за тренером моделей.
   * Обидва контури оцінюються паралельно: evaluate.py перевіряє Random Forest-розгалуження, а evaluate_xgb.py — XGBoost-розгалуження.
   * Завантажує лог прогнозів predictions_history.csv та predictions_history_xgb.csv, групує запити до Yahoo Finance для збору фактичних цільових цін закриття (Ground Truth).
   * Рахує математичну похибку моделей (MAE у відсотках та доларах) та Directional Accuracy (Win Rate) для обох часових горизонтів (1d та 5d).
   * Забезпечує абсолютну ідемпотентність: маркує перевірені сутності унікальним композитним ключем, виключаючи дублювання перевірок.
   * Захищає від вихідних днів та біржових свят, рахуючи реальні торгові сесії, що відбулися після дати прогнозу, замість наївного зсуву через BusinessDay.
   * Автоматично версіонує розширену матрицю метрик в evaluation_history.csv на Hugging Face Datasets та надсилає деталізований репорт у Telegram, розбитий по рядках на чанки по 3800 символів.

4. Public Client Inference (predict.py):
   * Легковажний скрипт для кінцевих користувачів або сторонніх сервісів (on-demand інференс).
   * Працює без токенів: стягує останні зафіксовані ваги моделей з Hugging Face Hub, локально прораховує аналогічний математичний граф фіч для поточної дати за останні 2 роки історії (period="2y") і миттєво виводить подвійний прогноз у консоль.

---

## 📊 Матриця вхідних ознак (16 Feature Columns)

Для ухвалення рішень модель використовує збалансований стек ознак:

| Категорія | Назва фічі | Опис індикатора |
| :--- | :--- | :--- |
| Технічні базові | `Close`, `Volume` | Поточна ціна закриття та об'єм торгів |
|  | `MA_5`, `MA_20` | Короткострокові ковзні середні (тиждень та місяць) |
|  | `Daily_Return`, `Volatility_5` | Добова доходність та рівень нервозності ринку за 5 днів |
| Структура сесії | `Intraday_Return`, `Day_Range` | Рух ціни всередині сесії та амплітуда (High/Low) торгів |
|  | `Gap` | Розмір нічного стрибка ціни (розрив відкриття) |
| Календарні | `Day_of_Week` | День тижня для врахування «п'ятничних фіксацій» прибутку |
| Об'єми торгів | `Volume_Ratio` | Сплеск торгів (поточний об'єм відносно 15-денного середнього) |
| Макро-контекст | `SP500_Return`, `VIX_Close` | Доходність індексу S&P 500 та рівень паніки Волл-стріт |
| Моментум | `RSI_14` | Індекс відносної сили для детекції зон перегріву |
|  | `Distance_to_MA200` | Відстань ціни від глобального річного тренду (200-денна середня) |
| Фундаментал | `Earnings_Season` | Прапорець сезону квартальних звітів корпорацій (1, 4, 7, 10 місяці) |
|  | `PE_Ratio`, `PS_Ratio` | Динамічні щоденні коефіцієнти Price-to-Earnings та Price-to-Sales |
|  | `Revenue_Growth` | Швидкість масштабування бізнесу за останніми даними компанії |

---

## 🔧 Конфігурація та змінні оточення

Система є повністю stateless і гнучко масштабується без необхідності перезбірки Docker-образу:

* STOCK_TICKER — список тікерів через кому (наприклад: NVDA,GOOG,AAPL,MSFT,ASML,TSM).
* HF_TOKEN — токен доступу до Hugging Face з правами Write.
* HF_REPO — репозиторій датасетів (username/predictive-stock-dataset).
* HF_MODEL_REPO — репозиторій моделей (username/predictive-stock-models).
* HF_MODEL_REPO_XGB — репозиторій ваг XGBoost-моделей (nadtoka/predictive-stock-models-xgb).
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

---

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

# Автоматичний аудит якості та розрахунок похибок моделей
docker run --rm \
  -e HF_TOKEN="your_token" \
  -e HF_REPO="nadtoka/predictive-stock-dataset" \
  -e TELEGRAM_BOT_TOKEN="your_tg_token" \
  -e TELEGRAM_CHAT_ID="your_tg_id" \
  -v /opt/stock-mlops/data:/app/data \
  ghcr.io/nadtoka/predictive-stock-mlops:latest python evaluate.py
```

---

## 📁 Структура проєкту

```
predictive-stock-mlops/
├── .github/workflows/
│   └── docker-ci.yml       # CI/CD пайплайн (Build без кешу -> Smoke Test -> Push)
├── data/                   # Кеш історії котирувань та індексів (ігнорується Git)
├── models/                 # Кеш ваг моделей Random Forest (.joblib, ігнорується Git)
├── models_xgb/             # Кеш ваг моделей XGBoost (.joblib, ігнорується Git)
├── Dockerfile              # Універсальна інструкція збірки імутабельного середовища
├── fetch_data.py           # Модуль збору даних (Акції + S&P 500 + VIX)
├── run_pipeline.sh         # Головний оркестратор A/B-пайплайну
├── train.py                # Контур A Random Forest, збір 16 фіч, подвійна валідація MAE, Telegram-репорт
├── train_xgb.py            # Контур B XGBoost, multi-output навчання, Telegram-репорт
├── evaluate.py             # Автоматичний аудит точності Random Forest, розрахунок MAE % та Win Rate
├── evaluate_xgb.py         # Автоматичний аудит точності XGBoost та зворотний зв'язок у продакшні
├── predict.py              # Публічний інференс на 2 роки історії (прогноз на 1д та 5д)
└── requirements.txt        # Фіксовані версії бібліотек
```
---

## Інтерфейс Telegram бота

![Telegram Operational Report](https://github.com/user-attachments/assets/ff329bf4-1e32-4291-b358-413924483988)
