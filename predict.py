import os
import warnings
import joblib
import pandas as pd
import yfinance as yf

# Глушимо внутрішні FutureWarning від yfinance на нових версіях Python/Pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

from huggingface_hub import hf_hub_download


def load_evaluation_history():
    """Завантажує історію оцінок для Continuous Learning без обов'язкових токенів"""
    hf_repo = os.getenv("HF_REPO", "nadtoka/predictive-stock-dataset")
    eval_url = f"https://huggingface.co/datasets/{hf_repo}/raw/main/evaluation_history.csv"
    try:
        df_eval = pd.read_csv(eval_url)
        if not df_eval.empty and "actual_price" in df_eval.columns:
            df_eval = df_eval.dropna(subset=["actual_price", "mae_pct"])
            df_eval = df_eval[df_eval["actual_price"] > 0]
        return df_eval
    except Exception as e:
        print(f"ℹ️ Не вдалося завантажити evaluation_history.csv (працюємо без компенсації зсуву): {e}")
        return pd.DataFrame()


def calculate_ticker_metrics(df_eval, current_ticker):
    """
    Розрахунок метрик для тікера на основі останніх 14 календарних днів.
    Повертає bias = 0.0, якщо перевірених оцінок недостатньо (< 5).
    """
    metrics = {
        "1d": {"win_rate": 50.0, "bias": 0.0, "count": 0},
        "5d": {"win_rate": 50.0, "bias": 0.0, "count": 0},
        "20d": {"win_rate": 50.0, "bias": 0.0, "count": 0},
    }
    if df_eval is None or df_eval.empty or "ticker" not in df_eval.columns:
        return metrics

    df_ticker = df_eval[df_eval["ticker"] == current_ticker].copy()
    if df_ticker.empty or "target_date" not in df_ticker.columns:
        return metrics

    df_ticker["target_date"] = pd.to_datetime(df_ticker["target_date"], utc=True).dt.tz_localize(None)
    now_date = pd.Timestamp.now().tz_localize(None).floor("D")
    cutoff_date = now_date - pd.Timedelta(days=14)

    for horizon in ["1d", "5d", "20d"]:
        df_h = df_ticker[df_ticker["horizon"] == horizon].copy()
        if df_h.empty:
            continue

        df_h = df_h.sort_values("target_date")
        df_recent = df_h[df_h["target_date"] >= cutoff_date]

        if len(df_recent) < 5:
            df_recent = df_h.tail(5)

        if len(df_recent) >= 5:
            count = len(df_recent)
            win_rate = 50.0
            if "direction_correct" in df_recent.columns and not df_recent["direction_correct"].dropna().empty:
                win_rate = float(df_recent["direction_correct"].mean() * 100)

            bias = 0.0
            if "actual_price" in df_recent.columns and "predicted_price" in df_recent.columns:
                diff = df_recent["actual_price"] - df_recent["predicted_price"]
                if not diff.dropna().empty:
                    bias = float(diff.median())

            metrics[horizon] = {
                "win_rate": win_rate if not pd.isna(win_rate) else 50.0,
                "bias": bias if not pd.isna(bias) else 0.0,
                "count": count,
            }

    return metrics


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
    target_mean_price = info.get("targetMeanPrice")
    forward_pe = info.get("forwardPE")
    trailing_pe = info.get("trailingPE")
    recommendation_mean = info.get("recommendationMean")

    df = stock.history(period="2y")

    if df.empty:
        raise ValueError(f"Не вдалося отримати дані для {ticker}")

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

    df = df.join(sp500_df[["SP500_Return"]], how="left")
    df = df.join(vix_df[["VIX_Close"]], how="left")

    df.loc[:, "MA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df.loc[:, "MA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df.loc[:, "Daily_Return"] = df["Close"].pct_change(fill_method=None)
    df.loc[:, "Volatility_5"] = df["Daily_Return"].rolling(window=5, min_periods=1).std().fillna(0)
    df.loc[:, "Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df.loc[:, "Day_Range"] = (df["High"] - df["Low"]) / df["Low"]
    df.loc[:, "Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df.loc[:, "Day_of_Week"] = df.index.dayofweek
    df.loc[:, "Volume_MA15"] = df["Volume"].rolling(window=15, min_periods=1).mean()
    df.loc[:, "Volume_Ratio"] = df["Volume"] / df["Volume_MA15"]
    df.loc[:, "MA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    df.loc[:, "Distance_to_MA200"] = (df["Close"] - df["MA_200"]) / df["MA_200"]
    df.loc[:, "Month"] = df.index.month
    df.loc[:, "Earnings_Season"] = df["Month"].isin([1, 4, 7, 10]).astype(int)
    df.loc[:, "PE_Ratio"] = df["Close"] / eps
    df.loc[:, "PS_Ratio"] = df["Close"] / rev_per_share
    df.loc[:, "Revenue_Growth"] = rev_growth
    if target_mean_price is not None and target_mean_price > 0:
        df.loc[:, "Analyst_Upside"] = (target_mean_price - df["Close"]) / df["Close"]
    else:
        df.loc[:, "Analyst_Upside"] = 0.0

    if (
        forward_pe is not None
        and trailing_pe is not None
        and forward_pe > 0
        and trailing_pe > 0
    ):
        df.loc[:, "PE_Expansion"] = float(forward_pe / trailing_pe)
    else:
        df.loc[:, "PE_Expansion"] = 1.0

    if recommendation_mean is not None and 1.0 <= recommendation_mean <= 5.0:
        df.loc[:, "Analyst_Score"] = float(recommendation_mean)
    else:
        df.loc[:, "Analyst_Score"] = 2.5

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df.loc[:, "RSI_14"] = 100 - (100 / (1 + rs))

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
        "Analyst_Upside",
        "PE_Expansion",
        "Analyst_Score",
    ]

    df_latest = df.dropna(subset=feature_cols)

    if df_latest.empty or len(df_latest) < 25:
        raise ValueError(f"❌ Недостатньо даних для розрахунку індикаторів для {ticker}.")

    return df_latest[feature_cols].tail(1)


def run_inference(ticker, df_eval=None):
    try:
        print(f"\n📦 Завантаження моделі {ticker} з Hugging Face Hub (nadtoka)...")
        model_file_path = hf_hub_download(
            repo_id="nadtoka/predictive-stock-models",
            filename=f"{ticker}_model.joblib",
            repo_type="model",
        )

        model = joblib.load(model_file_path)
        latest_features = prepare_features_for_prediction(ticker)
        current_price = latest_features["Close"].values[0]
        vix_current = latest_features["VIX_Close"].values[0] if "VIX_Close" in latest_features else 15.0

        preds = model.predict(latest_features)[0]
        tomorrow_pred = current_price * (1 + preds[0])
        week_pred = current_price * (1 + preds[1])
        month_pred = current_price * (1 + preds[2])

        # 🧠 CONTINUOUS LEARNING: Корекція зсуву та визначення бейджа надійності
        ticker_metrics = calculate_ticker_metrics(df_eval, ticker)
        bias_1d = ticker_metrics["1d"]["bias"]
        bias_5d = ticker_metrics["5d"]["bias"]
        bias_20d = ticker_metrics["20d"]["bias"]
        count_1d = ticker_metrics["1d"]["count"]
        count_5d = ticker_metrics["5d"]["count"]
        count_20d = ticker_metrics["20d"]["count"]
        win_rate_1d = ticker_metrics["1d"]["win_rate"]

        # Застосовуємо компенсацію з демпфуванням 0.2 та лімітами (тільки якщо ринок стабільний: VIX <= 22)
        if vix_current <= 22.0:
            if count_1d >= 5:
                cap_1d = 0.005 * current_price  # ліміт ±0.5%
                capped_bias_1d = max(-cap_1d, min(bias_1d * 0.2, cap_1d))
                tomorrow_pred += capped_bias_1d

            if count_5d >= 5:
                cap_5d = 0.015 * current_price  # ліміт ±1.5%
                capped_bias_5d = max(-cap_5d, min(bias_5d * 0.2, cap_5d))
                week_pred += capped_bias_5d

            if count_20d >= 5:
                cap_20d = 0.030 * current_price
                capped_bias_20d = max(-cap_20d, min(bias_20d * 0.2, cap_20d))
                month_pred += capped_bias_20d

        # Бейдж надійності
        if count_1d < 5:
            badge = "🟡"
        elif win_rate_1d >= 65.0:
            badge = "🟢"
        elif win_rate_1d >= 45.0:
            badge = "🟡"
        else:
            badge = "🔴"

        last_date = latest_features.index[-1]
        ua_days = {0: "Понеділок", 1: "Вівторок", 2: "Середа", 3: "Четвер", 4: "П'ятниця", 5: "Субота", 6: "Неділя"}

        date_1d = last_date + pd.offsets.BusinessDay(1)
        date_5d = last_date + pd.offsets.BusinessDay(5)
        date_20d = last_date + pd.offsets.BusinessDay(20)

        day_1d_text = f"{ua_days[date_1d.weekday()]} ({date_1d.strftime('%d.%m')})"
        day_5d_text = f"{ua_days[date_5d.weekday()]} ({date_5d.strftime('%d.%m')})"
        day_20d_text = f"{ua_days[date_20d.weekday()]} ({date_20d.strftime('%d.%m')})"

        emoji_1d = "📈" if tomorrow_pred > current_price else "📉"
        emoji_5d = "🚀" if week_pred > current_price else "📉"
        emoji_20d = "🚀" if month_pred > current_price else "📉"
        win_rate_str = f" [WR: {win_rate_1d:.0f}%]" if count_1d >= 5 else ""

        print(f"🔮 === ПРОГНОЗ ВІД ШІ ===")
        print(f"   📊 Тікер: {badge} {ticker}{win_rate_str}")
        print(f"   📈 Поточна ціна на ринку: ${current_price:.2f}")
        print(f"   🚀 Прогноз на {day_1d_text}: ${tomorrow_pred:.2f} {emoji_1d}")
        print(f"   📅 Прогноз на {day_5d_text}: ${week_pred:.2f} {emoji_5d}")
        print(f"   📆 Прогноз на {day_20d_text}: ${month_pred:.2f} {emoji_20d}")

    except Exception as e:
        print(f"❌ Сталася помилка під час інференсу для {ticker}: {e}")


if __name__ == "__main__":
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = list(dict.fromkeys(t.strip() for t in target_tickers.split(",") if t.strip()))

    print(f"🚀 Запуск інференсу для списку тікерів: {tickers}")
    df_eval_data = load_evaluation_history()

    for ticker in tickers:
        run_inference(ticker, df_eval=df_eval_data)