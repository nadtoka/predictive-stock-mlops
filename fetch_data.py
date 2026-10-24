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

    # 📅 Завантаження історичних та майбутніх дат квартальних звітів (Earnings Dates + EPS Surprise)
    try:
        ed = stock.get_earnings_dates(limit=28)
        if ed is not None and not ed.empty:
            ed.to_csv(f"data/{ticker}_earnings.csv")
            print(f"📅 Збережено календар звітів: data/{ticker}_earnings.csv ({len(ed)} дат)")
    except Exception as e:
        print(f"ℹ️ Календар звітів для {ticker} недоступний (пропускаємо): {e}")

    return True


def upload_to_huggingface():
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")  # Формат: "твій-юзернейм/назва-датасету"

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
            token=hf_token,
            ignore_patterns=["latest_predictions*.csv", "evaluation_history*local.csv"],
        )
        print("🔥 Датасет успішно оновлено на Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Помилка завантаження на Hugging Face: {e}")


if __name__ == "__main__":
    ticker_env = os.getenv("STOCK_TICKER", "AAPL")
    tickers = list(dict.fromkeys(t.strip() for t in ticker_env.split(",") if t.strip()))

    print(f"🚀 Запуск збору даних для списку: {tickers}")

    success_count = 0
    for ticker in tickers:
        if fetch_stock_data(ticker=ticker):
            success_count += 1

    # 🚀 Автоматичне завантаження макро-контексту та індикаторів волатильності
    print("📊 Завантаження макро-контексту: S&P 500 (^GSPC), VIX (^VIX), Nasdaq VXN (^VXN), US10Y (^TNX), Semiconductors (SOXX)...")
    macro_sources = [
        ("^GSPC", "data/SP500_history.csv", "S&P 500"),
        ("^VIX", "data/VIX_history.csv", "VIX"),
        ("^VXN", "data/VXN_history.csv", "Nasdaq Volatility (^VXN)"),
        ("^TNX", "data/TNX_history.csv", "US 10Y Treasury Yield (^TNX)"),
        ("SOXX", "data/SOXX_history.csv", "Semiconductor ETF (SOXX)"),
    ]
    for symbol, out_file, label in macro_sources:
        try:
            m_df = yf.Ticker(symbol).history(period="5y")
            if not m_df.empty:
                m_df.to_csv(out_file)
                print(f"✅ Дані {label} успішно збережено в {out_file}")
        except Exception as e:
            print(f"⚠️ Помилка завантаження {label} ({symbol}): {e}")

    print(f"🏁 Обробку завершено. Успішно зібрано: {success_count}/{len(tickers)}")

    # Викликаємо пуш на Hugging Face
    if success_count > 0:
        upload_to_huggingface()
