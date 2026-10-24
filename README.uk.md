# Predictive Stock MLOps Project 🚀 (v3.3)

🌐 **Мова / Language:** [🇬🇧 English](README.md) | [Українська]

Енд-ту-енд (End-to-End) MLOps проєкт для автоматизованого збору фінансових даних, щоденного перетренування прогнозних моделей (Continuous Training) під кожен актив окремо, версіонування артефактів у Hugging Face Hub та автоматичного моніторингу через Telegram. 

У версії **v3.3** система впроваджує **ШІ-Арбітр (`arbiter.py`)**, який зводить прогнози Random Forest і XGBoost у зважений консенсус-ансамбль і формує єдиний компактний Telegram-дайджест, а також переходить на **24 повністю стаціонарні (відносні) ознаки (`features.py`)** без сирих цін і без витоку майбутнього (з 20-денним embargo-буфером при валідації).

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
   * Глибокий Feature Engineering (Матриця з 20 ознак): Система динамічно збирає технічні, календарні, макроекономічні та фундаментальні фактори.
   * Multi-Output Навчання: Обидві моделі працюють у режимі багатоцільової регресії. Модель вчиться на відсоткових доходностях (pct_change) і за один прохід прогнозує вектор із трьох значень: рух на 1 день наперед (завтра), кумулятивний рух на 5 днів наперед (робочий тиждень) та на 20 днів наперед (місячний бізнес-тренд).
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
   * Рахує математичну похибку моделей (MAE у відсотках та доларах) та Directional Accuracy (Win Rate) для трьох часових горизонтів (1d, 5d та 20d).
   * Забезпечує абсолютну ідемпотентність: маркує перевірені сутності унікальним композитним ключем, виключаючи дублювання перевірок.
   * Захищає від вихідних днів та біржових свят, рахуючи реальні торгові сесії, що відбулися після дати прогнозу, замість наївного зсуву через BusinessDay.
   * Автоматично версіонує розширену матрицю метрик в evaluation_history.csv на Hugging Face Datasets та надсилає деталізований репорт у Telegram, розбитий по рядках на чанки по 3800 символів.

4. AI Arbiter & Ensemble Digest (arbiter.py):
   * Зводить історичний аудит обох моделей (Head-to-Head по 1d, 5d, 20d) і визначає персонального чемпіона для кожного тікера за останні 14 днів.
   * Розраховує зважений ансамбль прогнозів RF + XGBoost пропорційно їхньому Win Rate.
   * Замість 4 величезних повідомлень надсилає у Telegram один компактний, структурований дайджест: блок «ТОП Консенсус-сигнали» (де обидві моделі згодні 🤝 і WR ≥ 60%) та компактний список інших активів.

5. Public Client Inference (predict.py):
   * Легковажний скрипт для кінцевих користувачів або сторонніх сервісів (on-demand інференс).
   * Працює без токенів: стягує останні зафіксовані ваги обох моделей (RF та XGBoost) з Hugging Face Hub, будує граф із 24 стаціонарних фіч (`features.py`) і миттєво виводить консенсус-прогноз у консоль: **1д, 5д та 20д**.

---

## 📊 Матриця вхідних ознак (24 Stationary Feature Columns у `features.py`)

Для ухвалення рішень моделі використовують повністю стаціонарний (відносний) стек ознак без сирих цін та без витоку майбутнього:

| Категорія | Назва фічі | Опис індикатора |
| :--- | :--- | :--- |
| Імпульс доходності | `Daily_Return`, `Return_5d`, `Return_20d` | Відносна доходність за 1, 5 та 20 торгових сесій |
| Відстань до трендів | `Distance_to_MA5`, `Distance_to_MA20` | Відносне відхилення ціни від тижневої та місячної середньої |
| | `Distance_to_MA50`, `Distance_to_MA200` | Відносне відхилення від квартального та річного трендів |
| Волатильність | `Volatility_5`, `Volatility_20` | Короткострокова та місячна стандартна девіація доходностей |
| Структура сесії | `Intraday_Return`, `Day_Range`, `Gap` | Рух всередині сесії, амплітуда High/Low та нічний розрив відкриття |
| Календар та об'єм | `Day_of_Week`, `Earnings_Season`, `Volume_Ratio` | День тижня, сезон квартальних звітів та сплеск об'єму до MA15 |
| Макро та Альфа | `SP500_Return`, `Relative_Return_1d` | Доходність S&P 500 та чиста альфа акції відносно індексу |
| | `VIX_Close`, `VIX_Change` | Індекс страху Волл-стріт та його добовий імпульс |
| Осцилятори та шоки | `RSI_14`, `MACD_Hist`, `BB_Position` | Індекс відносної сили, нормалізований MACD та позиція в смугах Боллінджера |
| | `ATR_Ratio`, `ATR_Pct` | Шок діапазону (TR / ATR_14) та нормалізований ATR у % від ціни |

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
├── features.py             # Єдиний конвеєр 24 стаціонарних ознак, метрик та Telegram-утиліт
├── fetch_data.py           # Модуль збору даних (Акції + S&P 500 + VIX)
├── run_pipeline.sh         # Головний оркестратор A/B-пайплайну (Fetch -> Evaluate -> Train -> Arbiter)
├── train.py                # Контур A Random Forest, 20-денний embargo-буфер валідації, пуш на HF
├── train_xgb.py            # Контур B XGBoost, multi-output навчання, пуш на HF
├── evaluate.py             # Автоматичний аудит точності Random Forest з локального кешу data/
├── evaluate_xgb.py         # Автоматичний аудит точності XGBoost та зворотний зв'язок у продакшні
├── arbiter.py              # ШІ-Арбітр (RF vs XGBoost vs Ансамбль) та єдиний компактний дайджест у Telegram
├── predict.py              # Публічний ансамбль-інференс (прогноз на 1д, 5д та 20д)
└── requirements.txt        # Фіксовані версії бібліотек
```
---

## Інтерфейс Telegram бота

![Telegram Operational Report](https://github.com/user-attachments/assets/ff329bf4-1e32-4291-b358-413924483988)
