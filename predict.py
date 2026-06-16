import os
import yfinance as yf
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

def prepare_features_for_prediction(ticker):
    """
    Збір та підготовка свіжих ринкових даних для інференсу
    """
    print(f"📡 Збір свіжих даних з Yahoo Finance для {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo") # 1 місяця достатньо для розрахунку MA_20
    
    if df.empty:
        raise ValueError(f"Не вдалося отримати дані для {ticker}")
        
    df = df.copy()
    # Створюємо фічі (точно як при навчанні моделі)
    df.loc[:, 'MA_5'] = df['Close'].rolling(window=5).mean()
    df.loc[:, 'MA_20'] = df['Close'].rolling(window=20).mean()
    df.loc[:, 'Daily_Return'] = df['Close'].pct_change(fill_method=None)
    df.loc[:, 'Volatility_5'] = df['Daily_Return'].rolling(window=5).std()
    
    df_latest = df.dropna()
    feature_cols = ['Close', 'Volume', 'MA_5', 'MA_20', 'Daily_Return', 'Volatility_5']
    
    return df_latest[feature_cols].tail(1)

def run_inference(ticker):
    try:
        # 1. Завантажуємо ваги моделі з Hugging Face Model Registry
        print(f"📦 Завантаження моделі {ticker} з Hugging Face Hub (nadtoka)...")
        model_file_path = hf_hub_download(
            repo_id="nadtoka/predictive-stock-models", 
            filename=f"{ticker}_model.joblib",
            repo_type="model"
        )
        
        # 2. Завантажуємо модель в пам'ять
        model = joblib.load(model_file_path)
        
        # 3. Готуємо свіжі дані
        latest_features = prepare_features_for_prediction(ticker)
        current_price = latest_features['Close'].values[0]
        
        # 4. Робимо прогноз
        tomorrow_prediction = model.predict(latest_features)[0]
        
        # 5. Виводимо результат
        print(f"\n🔮 === ПРОГНОЗ ВІД ШІ ===")
        print(f"   📊 Тікер: {ticker}")
        print(f"   📈 Поточна ціна на ринку: ${current_price:.2f}")
        print(f"   🚀 Прогноз ціни на завтра: ${tomorrow_prediction:.2f}")
        
    except Exception as e:
        print(f"❌ Сталася помилка під час інференсу для {ticker}: {e}")

if __name__ == "__main__":
    # Можна передати один тікер через змінну оточення, або запустити дефолтний AAPL
    target_ticker = os.getenv("STOCK_TICKER", "AAPL")
    # Беремо перший тікер, якщо передано список через кому
    first_ticker = target_ticker.split(",")[0].strip()
    
    run_inference(ticker=first_ticker)
