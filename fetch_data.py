import os
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="AAPL", period="1y"):
    print(f"🚀 Починаємо завантаження даних для {ticker} за період {period}...")
    
    # Завантажуємо дані через yfinance
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        print(f"❌ Не вдалося отримати дані для тікера {ticker}")
        return
        
    # Створюємо папку для даних, якщо її немає
    os.makedirs("data", exist_ok=True)
    
    # Зберігаємо у CSV
    output_path = f"data/{ticker}_history.csv"
    df.to_csv(output_path)
    
    print(f"✅ Дані успішно збережено у: {output_path}")
    print(f"📊 Отримано рядків: {len(df)}")
    print(df.tail(3)) # Покажемо останні 3 дні для перевірки

if __name__ == "__main__":
    # Для тесту беремо Apple, але можна буде міняти через змінні оточення
    ticker_to_fetch = os.getenv("STOCK_TICKER", "AAPL")
    fetch_stock_data(ticker=ticker_to_fetch)
