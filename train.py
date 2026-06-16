import os
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from huggingface_hub import HfApi

def create_features(df):
    """
    Feature Engineering з виправленням Pandas FutureWarning
    """
    df = df.copy()
    df.loc[:, 'Target'] = df['Close'].shift(-1)
    df.loc[:, 'MA_5'] = df['Close'].rolling(window=5).mean()
    df.loc[:, 'MA_20'] = df['Close'].rolling(window=20).mean()
    df.loc[:, 'Daily_Return'] = df['Close'].pct_change(fill_method=None)
    df.loc[:, 'Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    return df.dropna()

def train_and_predict(ticker):
    data_path = f"data/{ticker}_history.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Дані для {ticker} не знайдені. Пропускаємо.")
        return False
        
    print(f"\n🍏 === Обробка тікера: {ticker} ===")
    df = pd.read_csv(data_path, index_col=0)
    
    df_prepared = create_features(df)
    
    feature_cols = ['Close', 'Volume', 'MA_5', 'MA_20', 'Daily_Return', 'Volatility_5']
    X = df_prepared[feature_cols]
    y = df_prepared['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"📊 Метрики валідації (історичні дані): MAE = ${mae:.2f}")
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{ticker}_model.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Модель збережено локально у: {model_path}")
    
    latest_data = X.tail(1)
    tomorrow_prediction = model.predict(latest_data)[0]
    current_price = latest_data['Close'].values[0]
    
    print(f"🔮 РЕЗУЛЬТАТ ПРЕДІКТУ ДЛЯ {ticker}:")
    print(f"   📈 Поточна ціна (остання відома): ${current_price:.2f}")
    print(f"   🚀 Прогноз ціни на наступний торговий день: ${tomorrow_prediction:.2f}")
    return True

def upload_models_to_huggingface():
    hf_token = os.getenv("HF_TOKEN")
    hf_model_repo = os.getenv("HF_MODEL_REPO") # Формат: "твій-юзернейм/назва-моделі"
    
    if not hf_token or not hf_model_repo:
        print("\nℹ️ HF_TOKEN або HF_MODEL_REPO не знайдені. Пропускаємо завантаження моделей на Hugging Face.")
        return

    print(f"\n🚀 Починаємо синхронізацію моделей з Hugging Face Model Registry: {hf_model_repo}...")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path="models",
            repo_id=hf_model_repo,
            repo_type="model",
            token=hf_token
        )
        print("🔥 Усі моделі успішно завантажено та заверсіоновано в Hugging Face Model Hub!")
    except Exception as e:
        print(f"❌ Помилка завантаження моделей на Hugging Face: {e}")

if __name__ == "__main__":
    target_tickers = os.getenv("STOCK_TICKER", "AAPL,MSFT,NVDA")
    tickers = [t.strip() for t in target_tickers.split(",") if t.strip()]
    
    print(f"🚀 Запуск тренування моделей для списку: {tickers}")
    
    success_count = 0
    for ticker in tickers:
        if train_and_predict(ticker):
            success_count += 1
            
    print(f"\n🏁 Тренування завершено. Успішно натреновано моделей: {success_count}/{len(tickers)}")
    
    if success_count > 0:
        upload_models_to_huggingface()
