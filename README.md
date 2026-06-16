# Predictive Stock MLOps Project #

Енд-ту-енд (End-to-End) MLOps проєкт для збору фінансових даних, навчання прогнозних моделей та побудови автоматизованого Pipeline для передбачення вартості акцій.

---

## 🏗️ Поточна архітектура (Крок 1)
На початковому етапі реалізовано модуль завантаження даних (Data Ingestion), який взаємодіє з Yahoo Finance API, агрегує історичні котирування та структурує їх за допомогою pandas.

---

## 🚀 Швидкий старт (Локальне розгортання)

### 1. Клонування репозиторію
git clone https://github.com/YOUR_GITHUB_USERNAME/predictive-stock-mlops.git
cd predictive-stock-mlops

### 2. Налаштування віртуального середовища
Рекомендується використовувати вбудований модуль venv для ізоляції залежностей Python.

**Для macOS / Linux:**
python3 -m venv venv
source venv/bin/activate

**Для Windows:**
python -m venv venv
venv\Scripts\activate

### 3. Встановлення залежностей
pip install -r requirements.txt

---

## 📊 Запуск збору даних

Для збору історичних котирувань запустіть скрипт fetch_data.py. За замовчуванням він завантажує дані для тікера AAPL (Apple Inc.) за останній рік:

python fetch_data.py

### Кастомізація через змінні оточення (Environment Variables)
Ви можете змінити цільовий актив без редагування коду за допомогою змінної STOCK_TICKER:

**Для macOS / Linux:**
export STOCK_TICKER="MSFT"
python fetch_data.py

**Для Windows:**
set STOCK_TICKER=MSFT
python fetch_data.py

Результати зберігаються локально в директорію data/ у форматі CSV. Ця папка автоматично додана до .gitignore для запобігання комміту сирих даних у репозиторій.

---

## 🛠️ Troubleshooting (Усунення несправностей)

### Помилка парсингу Yahoo Finance (Expecting value: line 1 column 1)
Якщо під час запуску скрипту ви бачите помилку:
"Failed to get ticker 'AAPL' reason: Expecting value: line 1 column 1 (char 0)"

Це означає, що Yahoo Finance заблокував запит через застарілі TLS/User-Agent відбитки бібліотеки. Для вирішення цієї проблеми оновіть yfinance до найсвіжішої версії, яка використовує curl_cffi для обходу блокувань:

pip install -U yfinance

---

## 📁 Структура проєкту
predictive-stock-mlops/
├── data/                  # Локальне сховище сирих даних (ігнорується Git)
│   └── AAPL_history.csv
├── venv/                  # Віртуальне оточення Python
├── .gitignore             # Конфігурація виключень Git
├── fetch_data.py          # Скрипт для стягування даних з Yahoo Finance
├── requirements.txt       # Залежності проєкту (yfinance, pandas)
└── README.md              # Документація проєкту
