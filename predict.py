import os
import warnings
import joblib
import pandas as pd
import yfinance as yf

# Глушимо внутрішні FutureWarning від yfinance на нових версіях Python/Pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

from huggingface_hub import hf_hub_download


def prepare_features_for_prediction(ticker):
    """Збір та підготовка свіжих ринкових даних для інференсу"""
    print(f"📡 Збір свіжих даних з Yahoo Finance для {ticker}...")
    stock = yf.Ticker(ticker)

    # 3 місяці історії для гарантованого прорахунку MA_20
    df = stock.history(period="3mo")

    if df.empty:
        raise ValueError(f"Не вдалося отримати дані для {ticker}")

    df = df.copy()
    # Створюємо фічі
    df.loc[:, "MA_5"] = df["Close"].rolling(window=5).mean()
    df.loc[:, "MA_20"] = df["Close"].rolling(window=20).mean()
    df.loc[:, "Daily_Return"] = df["Close"].pct_change(fill_method=None)
    df.loc[:, "Volatility_5"] = df["Daily_Return"].rolling(window=5).std()

    df_latest = df.dropna()

    if df_latest.empty:
        raise ValueError(
            f"❌ Недостатньо даних для розрахунку індикаторів для {ticker}."
        )

    feature_cols = [
        "Close",
        "Volume",
        "MA_5",
        "MA_20",
        "Daily_Return",
        "Volatility_5",
    ]
    return df_latest[feature_cols].tail(1)


def run_inference(ticker):
    try:
        # 1. Завантажуємо ваги моделі з Hugging Face
        print(
            f"\n📦 Завантаження моделі {ticker} з Hugging Face Hub (nadtoka)..."
        )
        model_file_path = hf_hub_download(
            repo_id="nadtoka/predictive-stock-models",
            filename=f"{ticker}_model.joblib",
            repo_type="model",
        )

        # 2. Завантажуємо модель в пам'ять
        model = joblib.load(model_file_path)

        # 3. Готуємо свіжі дані
        latest_features = prepare_features_for_prediction(ticker)
        current_price = latest_features["Close"].values[0]

        # 4. Робимо прогноз
        tomorrow_prediction = model.predict(latest_features)[0]

        # 5. Виводимо результат
        print(f"🔮 === ПРОГНОЗ ВІД ШІ ===")
        print(f"   📊 Тікер: {ticker}")
        print(f"   📈 Поточна ціна на ринку: ${current_price:.2f}")
        print(f"   🚀 Прогноз ціни на завтра: ${tomorrow_prediction:.2f}")

    except Exception as e:
        print(f"❌ Сталася помилка під час інференсу для {ticker}: {e}")


if __name__ == "__main__":
    # Парсимо список тікерів з енву (якщо порожньо — за замовчуванням AAPL)
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = [t.strip() for t in target_tickers.split(",") if t.strip()]

    print(f"🚀 Запуск інференсу для списку тікерів: {tickers}")

    for ticker in tickers:
        run_inference(ticker)
