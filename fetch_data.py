import os
import warnings
import yfinance as yf
import pandas as pd
from huggingface_hub import HfApi

warnings.simplefilter(action="ignore", category=FutureWarning)

def fetch_stock_data(ticker, period="5y"):
    print(f"📡 Завантаження даних для {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        print(f"❌ Не вдалося отримати дані для {ticker}")
        return False
        
    os.makedirs("data", exist_ok=True)
    output_path = f"data/{ticker}_history.csv"
    df.to_csv(output_path)
    print(f"✅ Збережено: {output_path} ({len(df)} рядків)")
    return True

def upload_to_huggingface():
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO") # Формат: "твій-юзернейм/назва-датасету"
    
    if not hf_token or not hf_repo:
        print("ℹ️ HF_TOKEN або HF_REPO не знайдені. Пропускаємо завантаження на Hugging Face.")
        return

    print(f"🚀 Починаємо синхронізацію датасету з Hugging Face репозиторієм: {hf_repo} ...")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path="data",
            repo_id=hf_repo,
            repo_type="dataset",
            token=hf_token
        )
        print("🔥 Датасет успішно оновлено на Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Помилка завантаження на Hug Hugging Face: {e}")

if __name__ == "__main__":
    ticker_env = os.getenv("STOCK_TICKER", "AAPL")
    tickers = [t.strip() for t in ticker_env.split(",") if t.strip()]
    
    print(f"🚀 Запуск збору даних для списку: {tickers}")
    
    success_count = 0
    for ticker in tickers:
        if fetch_stock_data(ticker=ticker):
            success_count += 1
    # 🚀  Автоматичне завантаження S&P 500
    print("📊 Завантаження макро-контексту: S&P 500 (^GSPC) та VIX (^VIX)...")
    try:
        # Завантажуємо S&P 500
        sp500 = yf.Ticker("^GSPC")
        sp500_df = sp500.history(period="5y")
        if not sp500_df.empty:
            sp500_df.to_csv("data/SP500_history.csv")
            print("✅ Дані S&P 500 успішно збережено в data/SP500_history.csv")
            
        # 🕵️ ЗАВАНТАЖУЄМО VIX
        vix = yf.Ticker("^VIX")
        vix_df = vix.history(period="5y")
        if not vix_df.empty:
            vix_df.to_csv("data/VIX_history.csv")
            print("✅ Дані VIX успішно збережено в data/VIX_history.csv")
         
    except Exception as e:
        print(f"❌ Помилка завантаження S&P 500: {e}")
            
    print(f"🏁 Обробку завершено. Успішно зібрано: {success_count}/{len(tickers)}")
    
    # Викликаємо пуш на Hugging Face
    if success_count > 0:
        upload_to_huggingface()
