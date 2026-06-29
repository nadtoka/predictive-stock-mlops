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
    info = {}
    try:
        info = stock.info
    except Exception:
        info = {}
    eps = info.get("trailingEps")
    eps = eps if eps not in (None, 0) else 1
    rev_per_share = info.get("revenuePerShare")
    rev_per_share = rev_per_share if rev_per_share not in (None, 0) else 1
    rev_growth = info.get("revenueGrowth")
    rev_growth = rev_growth if rev_growth not in (None, 0) else 0

    # 2 роки історії для гарантованого прорахунку MA_200
    df = stock.history(period="2y")

    if df.empty:
        raise ValueError(f"Не вдалося отримати дані для {ticker}")

    # Завантажуємо макро-індикатори, як у train.py
    sp500 = yf.Ticker("^GSPC")
    vix = yf.Ticker("^VIX")

    sp500_df = sp500.history(period="2y")
    vix_df = vix.history(period="2y")

    if sp500_df.empty:
        raise ValueError("Не вдалося отримати дані S&P 500 для інференсу")
    if vix_df.empty:
        raise ValueError("Не вдалося отримати дані VIX для інференсу")

    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).normalize()
    sp500_df.index = pd.to_datetime(sp500_df.index, utc=True).normalize()
    vix_df.index = pd.to_datetime(vix_df.index, utc=True).normalize()

    sp500_df["SP500_Return"] = sp500_df["Close"].pct_change(fill_method=None)
    vix_df["VIX_Close"] = vix_df["Close"]

    # Приєднуємо макро-фічі до основної таблиці за датою
    df = df.join(sp500_df[["SP500_Return"]], how="left")
    df = df.join(vix_df[["VIX_Close"]], how="left")

    # Створюємо фічі, максимально наближені до тієї самої матриці, що в train.py
    df.loc[:, "MA_5"] = df["Close"].rolling(window=5).mean()
    df.loc[:, "MA_20"] = df["Close"].rolling(window=20).mean()
    df.loc[:, "Daily_Return"] = df["Close"].pct_change(fill_method=None)
    df.loc[:, "Volatility_5"] = df["Daily_Return"].rolling(window=5).std()
    df.loc[:, "Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df.loc[:, "Day_Range"] = (df["High"] - df["Low"]) / df["Low"]
    df.loc[:, "Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df.loc[:, "Day_of_Week"] = df.index.dayofweek
    df.loc[:, "Volume_MA15"] = df["Volume"].rolling(window=15).mean()
    df.loc[:, "Volume_Ratio"] = df["Volume"] / df["Volume_MA15"]
    df.loc[:, "MA_200"] = df["Close"].rolling(window=200).mean()
    df.loc[:, "Distance_to_MA200"] = (df["Close"] - df["MA_200"]) / df["MA_200"]
    df.loc[:, "Month"] = df.index.month
    df.loc[:, "Earnings_Season"] = df["Month"].isin([1, 4, 7, 10]).astype(int)
    df.loc[:, "PE_Ratio"] = df["Close"] / eps
    df.loc[:, "PS_Ratio"] = df["Close"] / rev_per_share
    df.loc[:, "Revenue_Growth"] = rev_growth

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df.loc[:, "RSI_14"] = 100 - (100 / (1 + rs))

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
        "Intraday_Return",
        "Day_Range",
        "Gap",
        "Day_of_Week",
        "Volume_Ratio",
        "SP500_Return",
        "VIX_Close",
        "RSI_14",
        "Distance_to_MA200",
        "Earnings_Season",
        "PE_Ratio",
        "PS_Ratio",
        "Revenue_Growth",
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

        # 4. Робимо прогноз і перетворюємо відсотки у USD
        preds = model.predict(latest_features)[0]
        tomorrow_pred = current_price * (1 + preds[0])
        week_pred = current_price * (1 + preds[1])

        # 5. Виводимо результат
        print(f"🔮 === ПРОГНОЗ ВІД ШІ ===")
        print(f"   📊 Тікер: {ticker}")
        print(f"   📈 Поточна ціна на ринку: ${current_price:.2f}")
        print(f"   🚀 Прогноз ціни на завтра: ${tomorrow_pred:.2f}")
        print(f"   📅 Прогноз ціни через тиждень: ${week_pred:.2f}")

    except Exception as e:
        print(f"❌ Сталася помилка під час інференсу для {ticker}: {e}")


if __name__ == "__main__":
    # Парсимо список тікерів з енву (якщо порожньо — за замовчуванням AAPL)
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = [t.strip() for t in target_tickers.split(",") if t.strip()]

    print(f"🚀 Запуск інференсу для списку тікерів: {tickers}")

    for ticker in tickers:
        run_inference(ticker)
