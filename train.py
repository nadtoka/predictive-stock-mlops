import os
import warnings
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from curl_cffi import requests
from huggingface_hub import HfApi

# Глушимо FutureWarning від Pandas/yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

def send_telegram_report(text):
    """Надсилає фінальний аналітичний звіт в Телеграм чат"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("ℹ️ TELEGRAM_BOT_TOKEN або TELEGRAM_CHAT_ID не знайдені. Пропускаємо сповіщення.")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("🚀 Аналітичний звіт успішно надіслано в Telegram!")
        else:
            print(f"❌ Помилка Telegram API: {res.text}")
    except Exception as e:
        print(f"❌ Не вдалося зв'язатися з Telegram: {e}")

def train_and_upload():
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = [t.strip() for t in target_tickers.split(",") if t.strip()]
    
    hf_token = os.getenv("HF_TOKEN")
    hf_model_repo = os.getenv("HF_MODEL_REPO")
    
    os.makedirs("models", exist_ok=True)
    
    # Заголовок нашого щонічного звіту
    tg_report = "📊 *ЩОНІЧНИЙ MLOps ЗВІТ ШІ*\n"
    tg_report += "━━━━━━━━━━━━━━━━━━━━\n\n"
    
    successful_models = 0

    for ticker in tickers:
        data_path = f"data/{ticker}_history.csv"
        if not os.path.exists(data_path):
            print(f"⚠️ Файл даних для {ticker} не знайдено. Пропускаємо.")
            continue
            
        print(f"🧠 Обробка та тренування моделі для {ticker}...")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)        
        df.index = pd.to_datetime(df.index,utc=True).normalize()

	# 🚀  ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ S&P 500
        sp500_path = "data/SP500_history.csv"
        if os.path.exists(sp500_path):
            sp500_df = pd.read_csv(sp500_path, index_col=0, parse_dates=True)
            sp500_df.index = pd.to_datetime(sp500_df.index, utc=True).normalize()
            
            # Рахуємо добову доходність усього ринку
            df['SP500_Return'] = sp500_df['Close'].pct_change(fill_method=None)
        else:
            print("⚠️ Файл S&P 500 не знайдено! Модель вчитиметься без макро-контексту.")
            df['SP500_Return'] = 0  # Заглушка, якщо файлу немає

	# 🚀  НОВЕ: ЗАВАНТАЖЕННЯ ТА ПІД КЛЕЙКА ІНДЕКСУ СТРАХУ VIX
        vix_path = "data/VIX_history.csv"
        if os.path.exists(vix_path):
            vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
            vix_df.index = pd.to_datetime(vix_df.index, utc=True).normalize()
            
            # Для VIX беремо чисте значення Close (рівень страху), а не відсоток зміни
            df['VIX_Close'] = vix_df['Close']
        else:
            print("⚠️ Файл VIX не знайдено! Використовуємо дефолтний спокійний рівень.")
            df['VIX_Close'] = 15.0  # Базова заглушка нормального ринку

        # Розрахунок технічних індикаторів (Фічі)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Daily_Return'] = df['Close'].pct_change(fill_method=None)
        df['Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
        
        df["Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]  # Рух всередині дня
        df["Day_Range"] = (df["High"] - df["Low"]) / df["Low"]  # Розмах торгів
        df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        
        df['Day_of_Week'] = df.index.dayofweek  # Понеділок = 0, П'ятниця = 4
        df['Volume_MA15'] = df['Volume'].rolling(window=15).mean() # Середня палата об'єму за 15 днів
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA15']     # Сплеск торгів (у скільки разів більший за норму)

        # ДОДАЄМО RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Запобігаємо діленню на нуль, якщо ринок стоїть на місці
        rs = gain / loss.replace(0, 1e-9) 
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Сначала удаляем NaN только из колонок ФИЧ (это очистит первые 20 строк от скользящих средних)
        df = df.dropna(subset=feature_cols)
        
        # Переносим сбор СВЕЖАЙШИХ фичей (пятницы) СЮДА — до сдвига таргета!
        latest_features = df[feature_cols].tail(1).copy()
        current_price = latest_features['Close'].values[0]

        # Теперь создаем Target для обучения
        df['Target'] = df['Close'].shift(-1)
        
        # Удаляем НАЙДЕННЫЙ NaN только из целевой переменной (это корректно удалит пятницу ИЗ ОБУЧАЮЩЕЙ выборки)
        df = df.dropna(subset=['Target'])

        
        if df.empty:
            print(f"❌ Недостатньо даних після створення індикаторів для {ticker}.")
            continue
            
        feature_cols = [
	    "Close",
	    "Volume",
	    "MA_5",
	    "MA_20",
	    "Daily_Return",
	    "Volatility_5",
	    "Intraday_Return",
	    "Day_Range",
	    "Gap",
            "Day_of_Week",
            "Volume_Ratio",
            "SP500_Return",
            "VIX_Close",
            "RSI_14"
        ]        
 
        # Спліт на Train/Test (Останні 20 днів для валідації метрик)
        train_df = df.iloc[:-20]
        test_df = df.iloc[-20:]
        
        X_train, y_train = train_df[feature_cols], train_df['Target']
        X_test, y_test = test_df[feature_cols], test_df['Target']
        
        # Навчання моделі
        # model = RandomForestRegressor(n_estimators=100, random_state=42)
        model = RandomForestRegressor(
    	n_estimators=200,      # Збільшили кількість дерев для стабільності
    	max_depth=12,          # Обмежили глибину, щоб не зубрила шуми
    	min_samples_leaf=3,    # Шукаємо загальні тренди, ігноруємо аномалії
    	# n_jobs=-1,             # ТУРБО-РЕЖИМ: паралелимо на всі ядра CPU (ТИМЧАСОВО НЕ ПРАЦЮЄ ЧЕРЕЗ БАГ СУМІСНОСТІ З ОСТАНЬО ВЕРСІЄЮ ПАЙТОН)
	random_state=42
	)
        model.fit(X_train, y_train)
        
        # Валідація
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        # Отримуємо фічі за СЬОГОДНІ для реального прогнозу на ЗАВТРА
        # latest_features = df[feature_cols].tail(1)
        # current_price = latest_features['Close'].values[0]
        tomorrow_prediction = model.predict(latest_features)[0]
        
        # Зберігаємо ваги індивідуальної моделі
        model_path = f"models/{ticker}_model.joblib"
        joblib.dump(model, model_path)
        successful_models += 1
        
        # Додаємо блок компанії у Телеграм-звіт
        trend_emoji = "📈" if tomorrow_prediction > current_price else "📉"
        tg_report += f"🔹 *{ticker}*:\n"
        tg_report += f"  • Поточна ціна: `${current_price:.2f}`\n"
        tg_report += f"  • Прогноз на завтра: `${tomorrow_prediction:.2f}` {trend_emoji}\n"
        tg_report += f"  • Похибка моделі (MAE): `${mae:.2f}`\n\n"

    # Спроба синхронізації з Hugging Face Model Registry
    if hf_token and hf_model_repo and successful_models > 0:
        try:
            print("📦 Пуш оновлених моделей на Hugging Face Hub...")
            api = HfApi()
            api.upload_folder(
                folder_path="models",
                repo_id=hf_model_repo,
                repo_type="model",
                token=hf_token
            )
            tg_report += "☁️ *Hugging Face:* Моделі успішно синхронізовано."
        except Exception as e:
            tg_report += f"⚠️ *Hugging Face:* Помилка завантаження ваг: {e}"
    else:
        tg_report += "ℹ️ *Hugging Face:* Синхронізацію пропущено (немає токенів)."

    # Відправляємо фінальний зібраний звіт в телеграм
    send_telegram_report(tg_report)

if __name__ == "__main__":
    train_and_upload()
