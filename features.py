import os
import warnings
import pandas as pd
import yfinance as yf
from curl_cffi import requests

warnings.simplefilter(action="ignore", category=FutureWarning)

# Список стаціонарних (відносних) ознак без сирих цін/об'ємів та без витоку майбутнього (look-ahead bias).
# Всі фічі нормалізовані у відсотках або коефіцієнтах, щоб дерева рішень (Random Forest та XGBoost)
# однаково точно працювали як на історії 5-річної давнини, так і на поточних максимумах ціни.
FEATURE_COLS = [
    "Daily_Return",
    "Return_5d",
    "Return_20d",
    "Distance_to_MA5",
    "Distance_to_MA20",
    "Distance_to_MA50",
    "Distance_to_MA200",
    "Volatility_5",
    "Volatility_20",
    "Intraday_Return",
    "Day_Range",
    "Gap",
    "Day_of_Week",
    "Volume_Ratio",
    "SP500_Return",
    "Relative_Return_1d",
    "VIX_Close",
    "VIX_Change",
    "RSI_14",
    "MACD_Hist",
    "BB_Position",
    "ATR_Ratio",
    "ATR_Pct",
    "Earnings_Season",
]


def send_telegram_report(text, max_len=3800):
    """Надсилає звіт в Телеграм із розбиттям на безпечні чанки за рядками та fallback без Markdown."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ℹ️ TELEGRAM_BOT_TOKEN або TELEGRAM_CHAT_ID не знайдені. Пропускаємо сповіщення.")
        return

    chunks = []
    current_chunk = ""

    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > max_len:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for idx, chunk in enumerate(chunks, 1):
        if not chunk:
            continue
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown",
        }
        try:
            res = requests.post(url, json=payload, timeout=15)
            if res.status_code == 200:
                print(f"🚀 Частину {idx}/{len(chunks)} успішно надіслано в Telegram!")
            else:
                payload.pop("parse_mode", None)
                res_plain = requests.post(url, json=payload, timeout=15)
                if res_plain.status_code == 200:
                    print(f"🚀 Частину {idx}/{len(chunks)} надіслано звичайним текстом (fallback)!")
                else:
                    print(f"❌ Помилка Telegram API (частина {idx}): {res.text}")
        except Exception as e:
            print(f"❌ Не вдалося зв'язатися з Telegram: {e}")


def build_features(df, sp500_df=None, vix_df=None):
    """
    Єдиний конвеєр розрахунку 24 стаціонарних технічних, макро- та календарних ознак.
    Використовує еластичні ковзні вікна (min_periods=1) для підтримки молодих IPO-активів.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).normalize()

    # Макро-контекст: S&P 500
    if sp500_df is not None and not sp500_df.empty and "Close" in sp500_df:
        sp_copy = sp500_df.copy()
        sp_copy.index = pd.to_datetime(sp_copy.index, utc=True).normalize()
        sp_ret = sp_copy["Close"].pct_change(fill_method=None)
        df["SP500_Return"] = sp_ret.reindex(df.index).ffill().fillna(0.0)
    elif "SP500_Return" not in df.columns:
        df["SP500_Return"] = 0.0

    # Макро-контекст: VIX
    if vix_df is not None and not vix_df.empty and "Close" in vix_df:
        vx_copy = vix_df.copy()
        vx_copy.index = pd.to_datetime(vx_copy.index, utc=True).normalize()
        df["VIX_Close"] = vx_copy["Close"].reindex(df.index).ffill().fillna(15.0)
    elif "VIX_Close" not in df.columns:
        df["VIX_Close"] = 15.0

    # Базові доходності на різних часових масштабах
    df["Daily_Return"] = df["Close"].pct_change(fill_method=None)
    df["Return_5d"] = df["Close"].pct_change(periods=5, fill_method=None).fillna(df["Daily_Return"])
    df["Return_20d"] = df["Close"].pct_change(periods=20, fill_method=None).fillna(df["Return_5d"])

    # Відносні відстані до ковзних середніх (замість сирих цін MA_5, MA_20, Close)
    ma_5 = df["Close"].rolling(window=5, min_periods=1).mean()
    ma_20 = df["Close"].rolling(window=20, min_periods=1).mean()
    ma_50 = df["Close"].rolling(window=50, min_periods=1).mean()
    ma_200 = df["Close"].rolling(window=200, min_periods=1).mean()

    df["Distance_to_MA5"] = ((df["Close"] - ma_5) / ma_5.replace(0, 1e-9)).fillna(0.0)
    df["Distance_to_MA20"] = ((df["Close"] - ma_20) / ma_20.replace(0, 1e-9)).fillna(0.0)
    df["Distance_to_MA50"] = ((df["Close"] - ma_50) / ma_50.replace(0, 1e-9)).fillna(0.0)
    df["Distance_to_MA200"] = ((df["Close"] - ma_200) / ma_200.replace(0, 1e-9)).fillna(0.0)

    # Волатильність (короткострокова та місячна)
    df["Volatility_5"] = df["Daily_Return"].rolling(window=5, min_periods=1).std().fillna(0.0)
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20, min_periods=1).std().fillna(df["Volatility_5"])

    # Внутрішньоденна структура сесії та нічні гепи
    df["Intraday_Return"] = ((df["Close"] - df["Open"]) / df["Open"].replace(0, 1e-9)).fillna(0.0)
    df["Day_Range"] = ((df["High"] - df["Low"]) / df["Low"].replace(0, 1e-9)).fillna(0.0)
    prev_close = df["Close"].shift(1)
    df["Gap"] = ((df["Open"] - prev_close) / prev_close.replace(0, 1e-9)).fillna(0.0)

    # Календарні та об'ємні фактори
    df["Day_of_Week"] = df.index.day_of_week
    vol_ma15 = df["Volume"].rolling(window=15, min_periods=1).mean()
    df["Volume_Ratio"] = (df["Volume"] / vol_ma15.replace(0, 1e-9)).fillna(1.0)

    # Відносна сила акції проти ринку (Alpha) та динаміка страху VIX
    df["Relative_Return_1d"] = (df["Daily_Return"] - df["SP500_Return"]).fillna(0.0)
    df["VIX_Change"] = df["VIX_Close"].pct_change(fill_method=None).fillna(0.0)

    # Сезон квартальних звітів
    df["Month"] = df.index.month
    df["Earnings_Season"] = df["Month"].isin([1, 4, 7, 10]).astype(int)

    # ATR (Average True Range) — шоковий імпульс та відносна амплітуда
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=1).mean()
    df["ATR_Ratio"] = (tr / atr_14.replace(0, 1e-9)).fillna(1.0)
    df["ATR_Pct"] = (atr_14 / df["Close"].replace(0, 1e-9)).fillna(0.0)

    # RSI 14
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=2).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=2).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # Нормалізований MACD Histogram (відносно ціни закриття)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_pct = (ema12 - ema26) / df["Close"].replace(0, 1e-9)
    macd_sig = macd_pct.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = (macd_pct - macd_sig).fillna(0.0)

    # Позиція ціни всередині смуг Боллінджера (Bollinger Band Z-Score)
    bb_std = df["Close"].rolling(window=20, min_periods=2).std().fillna(0.0)
    df["BB_Position"] = ((df["Close"] - ma_20) / (2.0 * bb_std.replace(0, 1e-9))).clip(-3.0, 3.0).fillna(0.0)

    return df


def calculate_ticker_metrics(df_eval, current_ticker):
    """
    Уніфікований розрахунок метрик точності та відносного зсуву (bias_pct) для тікера
    на основі останніх 14 календарних днів.
    Використовує однакові чесні правила як для Random Forest, так і для XGBoost:
    якщо за останні 14 днів менше 5 свіжих перевірок — компенсація вимикається (bias = 0),
    щоб не тягнути застарілий хвіст із минулого місяця.
    """
    metrics = {
        "1d": {"win_rate": 50.0, "bias": 0.0, "bias_pct": 0.0, "mae_pct": 0.0, "count": 0},
        "5d": {"win_rate": 50.0, "bias": 0.0, "bias_pct": 0.0, "mae_pct": 0.0, "count": 0},
        "20d": {"win_rate": 50.0, "bias": 0.0, "bias_pct": 0.0, "mae_pct": 0.0, "count": 0},
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
            metrics[horizon] = {
                "win_rate": 50.0,
                "bias": 0.0,
                "bias_pct": 0.0,
                "mae_pct": float(df_h["mae_pct"].tail(10).mean()) if "mae_pct" in df_h and not df_h.empty else 0.0,
                "count": int(len(df_recent)),
            }
            continue

        count = int(len(df_recent))
        win_rate = 50.0
        if "direction_correct" in df_recent.columns and not df_recent["direction_correct"].dropna().empty:
            win_rate = float(df_recent["direction_correct"].mean() * 100.0)

        bias = 0.0
        bias_pct = 0.0
        if "actual_price" in df_recent.columns and "predicted_price" in df_recent.columns:
            valid = df_recent.dropna(subset=["actual_price", "predicted_price"])
            valid = valid[valid["predicted_price"] > 0]
            if not valid.empty:
                diff = valid["actual_price"] - valid["predicted_price"]
                bias = float(diff.median())
                rel_diff = (valid["actual_price"] - valid["predicted_price"]) / valid["predicted_price"]
                bias_pct = float(rel_diff.median())

        mae_pct = 0.0
        if "mae_pct" in df_recent.columns and not df_recent["mae_pct"].dropna().empty:
            mae_pct = float(df_recent["mae_pct"].mean())

        metrics[horizon] = {
            "win_rate": 50.0 if pd.isna(win_rate) else win_rate,
            "bias": 0.0 if pd.isna(bias) else bias,
            "bias_pct": 0.0 if pd.isna(bias_pct) else bias_pct,
            "mae_pct": 0.0 if pd.isna(mae_pct) else mae_pct,
            "count": count,
        }

    return metrics


def apply_bias_correction(raw_price, current_price, ticker_metrics, horizon, vix_current, damping=0.2):
    """
    Динамічна компенсація зсуву у ВІДСОТКАХ (bias_pct) із захистом від паніки (VIX > 22).
    """
    caps = {"1d": 0.005, "5d": 0.015, "20d": 0.030}
    cap_pct = caps.get(horizon, 0.005)
    h_metrics = ticker_metrics.get(horizon, {})
    count = h_metrics.get("count", 0)
    bias_pct = h_metrics.get("bias_pct", 0.0)

    if count < 5 or vix_current > 22.0:
        return raw_price

    adj_pct = max(-cap_pct, min(bias_pct * damping, cap_pct))
    return raw_price * (1.0 + adj_pct)


def load_ground_truth_series(tickers, data_dir="data"):
    """
    Завантажує реальні історичні ціни закриття.
    Спочатку читає локальний кеш data/{ticker}_history.csv (5 років історії без зайвих мережевих запитів),
    а якщо файлу немає — робить безпечний fallback до yfinance (period='1y').
    """
    actual_data = {}
    for ticker in tickers:
        local_path = os.path.join(data_dir, f"{ticker}_history.csv")
        loaded = False
        if os.path.exists(local_path):
            try:
                stock_df = pd.read_csv(local_path, index_col=0, parse_dates=True)
                if not stock_df.empty and "Close" in stock_df:
                    stock_df = stock_df.dropna(subset=["Close"])
                    stock_df = stock_df[stock_df["Close"] > 0]
                    stock_df.index = pd.to_datetime(stock_df.index, utc=True).normalize()
                    actual_data[ticker] = stock_df["Close"]
                    loaded = True
            except Exception as e:
                print(f"⚠️ Помилка читання локального файлу {local_path}: {e}")

        if not loaded:
            try:
                stock_df = yf.Ticker(ticker).history(period="1y")
                if not stock_df.empty and "Close" in stock_df:
                    stock_df = stock_df.dropna(subset=["Close"])
                    stock_df = stock_df[stock_df["Close"] > 0]
                    stock_df.index = pd.to_datetime(stock_df.index, utc=True).normalize()
                    actual_data[ticker] = stock_df["Close"]
            except Exception as e:
                print(f"⚠️ Не вдалося отримати котирування для {ticker}: {e}")

    return actual_data
