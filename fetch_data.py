import os
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period="1y"):
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

if __name__ == "__main__":
    # Беремо список із екологічних змінних
    ticker_env = os.getenv("STOCK_TICKER", "AAPL")
    
    # Розбиваємо рядок на чисті тікери
    tickers = [t.strip() for t in ticker_env.split(",") if t.strip()]
    
    print(f"🚀 Запуск збору даних для списку: {tickers}")
    
    success_count = 0
    for ticker in tickers:
        if fetch_stock_data(ticker=ticker):
            success_count += 1
            
    print(f"🏁 Обробку завершено. Успішно зібрано: {success_count}/{len(tickers)}")
