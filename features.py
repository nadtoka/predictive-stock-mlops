import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests

warnings.simplefilter(action="ignore", category=FutureWarning)

# Список 32 стаціонарних (відносних) ознак без сирих цін/об'ємів та без витоку майбутнього (look-ahead bias).
# Включає технічні моментуми, макро-тригери ліквідності (^VXN, ^TNX, SOXX), структурну волатильність
# (Garman-Klass, Volatility Squeeze) та точний радар квартальних звітів (Earnings Proximity & Surprise).
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
    "GK_Volatility",
    "Volatility_Squeeze",
    "Intraday_Return",
    "Day_Range",
    "Gap",
    "Day_of_Week",
    "Volume_Ratio",
    "SP500_Return",
    "Relative_Return_1d",
    "VIX_Close",
    "VIX_Change",
    "VXN_Close",
    "VXN_VIX_Spread",
    "TNX_Change",
    "SOXX_Return",
    "Relative_to_SOXX",
    "RSI_14",
    "MACD_Hist",
    "BB_Position",
    "ATR_Ratio",
    "ATR_Pct",
    "Earnings_Season",
    "Earnings_Proximity",
    "Last_EPS_Surprise",
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


def load_macro_context(data_dir="data"):
    """Завантажує всі макро-індикатори з локального кешу data/ (S&P 500, VIX, VXN, TNX, SOXX)."""
    macro = {}
    mapping = {
        "sp500": "SP500_history.csv",
        "vix": "VIX_history.csv",
        "vxn": "VXN_history.csv",
        "tnx": "TNX_history.csv",
        "soxx": "SOXX_history.csv",
    }
    for key, filename in mapping.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            try:
                m_df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not m_df.empty and "Close" in m_df:
                    m_df.index = pd.to_datetime(m_df.index, utc=True).normalize()
                    macro[key] = m_df
            except Exception as e:
                print(f"⚠️ Помилка читання макро-файлу {path}: {e}")
    return macro


def load_earnings_calendar(ticker, data_dir="data"):
    """Завантажує календар квартальних звітів для тікера з локального кешу або порожній DataFrame."""
    path = os.path.join(data_dir, f"{ticker}_earnings.csv")
    if os.path.exists(path):
        try:
            ed = pd.read_csv(path, index_col=0, parse_dates=True)
            if not ed.empty:
                ed.index = pd.to_datetime(ed.index, utc=True).normalize()
                return ed
        except Exception:
            pass
    return None


def _apply_earnings_features(df, earnings_df):
    """
    Розраховує точні фактори волатильності навколо квартальних звітів (Earnings Call) без витоку майбутнього:
    - Earnings_Proximity: 1.0 у день звіту або за 1-5 торгових днів ДО нього (pre-earnings volatility run-up),
      -1.0 у перші 2 дні ПІСЛЯ звіту (post-earnings drift / IV crush), 0.0 в інші дні.
    - Last_EPS_Surprise: останній відомий сюрприз прибутку (Surprise(%)) на момент конкретної дати (через ffill).
    """
    df["Earnings_Proximity"] = 0.0
    df["Last_EPS_Surprise"] = 0.0

    if earnings_df is None or earnings_df.empty:
        return df

    try:
        ed = earnings_df.copy()
        ed.index = pd.to_datetime(ed.index, utc=True).normalize()
        ed = ed[~ed.index.duplicated(keep="first")].sort_index()

        # 1. Історичний сюрприз EPS (відомий ТІЛЬКИ після дати звіту -> без look-ahead bias)
        surprise_col = None
        for col in ed.columns:
            if "Surprise" in str(col):
                surprise_col = col
                break

        if surprise_col is not None:
            surp = pd.to_numeric(ed[surprise_col], errors="coerce").dropna() / 100.0
            if not surp.empty:
                # Reindex по об'єднаній осі та forward-fill, щоб кожен день знав лише МИНУЛИЙ звіт
                combined_idx = df.index.union(surp.index).sort_values()
                surp_aligned = surp.reindex(combined_idx).ffill().reindex(df.index).fillna(0.0)
                df["Last_EPS_Surprise"] = surp_aligned.clip(-2.0, 2.0)

        # 2. Близькість до дати звіту (Earnings_Proximity)
        event_dates = ed.index.values
        if len(event_dates) > 0:
            idx_vals = df.index.values
            # Для кожної дати знаходимо найближчий майбутній звіт та останній минулий звіт
            pos = np.searchsorted(event_dates, idx_vals)

            # Дні до наступного звіту (в календарних днях: <= 7 днів відповідає ~5 торговим сесіям)
            valid_next = pos < len(event_dates)
            days_to_next = np.full(len(idx_vals), 999.0)
            days_to_next[valid_next] = (
                (event_dates[pos[valid_next]] - idx_vals[valid_next]) / np.timedelta64(1, "D")
            )

            # Дні після минулого звіту
            valid_prev = pos > 0
            days_since_prev = np.full(len(idx_vals), 999.0)
            days_since_prev[valid_prev] = (
                (idx_vals[valid_prev] - event_dates[pos[valid_prev] - 1]) / np.timedelta64(1, "D")
            )

            prox = np.zeros(len(idx_vals), dtype=float)
            # Pre-earnings run-up: чим ближче до звіту (від 7 до 0 днів), тим вищий сигнал (від 0.3 до 1.0)
            pre_mask = (days_to_next >= 0) & (days_to_next <= 7)
            prox[pre_mask] = np.round(1.0 - (days_to_next[pre_mask] / 10.0), 2)

            # Post-earnings reaction (1-3 календарні дні після звіту): маркуємо як -0.5 (фаза IV crush / переоцінки)
            post_mask = (~pre_mask) & (days_since_prev >= 1) & (days_since_prev <= 3)
            prox[post_mask] = -0.5

            df["Earnings_Proximity"] = prox
    except Exception as e:
        print(f"ℹ️ Помилка обробки календаря звітів: {e}")

    return df


def build_features(
    df,
    sp500_df=None,
    vix_df=None,
    vxn_df=None,
    tnx_df=None,
    soxx_df=None,
    earnings_df=None,
):
    """
    Єдиний конвеєр розрахунку 33 стаціонарних технічних, макро-, волатильнісних та звітних ознак.
    Використовує еластичні ковзні вікна (min_periods=1) для підтримки молодих IPO-активів.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).normalize()

    # 1. Макро-контекст: S&P 500 (^GSPC)
    if sp500_df is not None and not sp500_df.empty and "Close" in sp500_df:
        sp_copy = sp500_df.copy()
        sp_copy.index = pd.to_datetime(sp_copy.index, utc=True).normalize()
        sp_ret = sp_copy["Close"].pct_change(fill_method=None)
        df["SP500_Return"] = sp_ret.reindex(df.index).ffill().fillna(0.0)
    elif "SP500_Return" not in df.columns:
        df["SP500_Return"] = 0.0

    # 2. Макро-контекст: VIX (^VIX)
    if vix_df is not None and not vix_df.empty and "Close" in vix_df:
        vx_copy = vix_df.copy()
        vx_copy.index = pd.to_datetime(vx_copy.index, utc=True).normalize()
        df["VIX_Close"] = vx_copy["Close"].reindex(df.index).ffill().fillna(15.0)
    elif "VIX_Close" not in df.columns:
        df["VIX_Close"] = 15.0

    # 3. Макро-волатильність тех-сектору: Nasdaq Volatility (^VXN)
    if vxn_df is not None and not vxn_df.empty and "Close" in vxn_df:
        vxn_copy = vxn_df.copy()
        vxn_copy.index = pd.to_datetime(vxn_copy.index, utc=True).normalize()
        df["VXN_Close"] = vxn_copy["Close"].reindex(df.index).ffill().fillna(df["VIX_Close"] * 1.2)
    else:
        df["VXN_Close"] = df["VIX_Close"] * 1.2

    # Спред паніки Nasdaq проти широкого ринку (коли тех-сектор штормить сильніше за S&P 500)
    df["VXN_VIX_Spread"] = ((df["VXN_Close"] - df["VIX_Close"]) / df["VIX_Close"].replace(0, 1e-9)).fillna(0.0)

    # 4. Макро-тригер вартості грошей: US 10-Year Treasury Yield (^TNX)
    if tnx_df is not None and not tnx_df.empty and "Close" in tnx_df:
        tnx_copy = tnx_df.copy()
        tnx_copy.index = pd.to_datetime(tnx_copy.index, utc=True).normalize()
        tnx_ret = tnx_copy["Close"].pct_change(fill_method=None)
        df["TNX_Change"] = tnx_ret.reindex(df.index).ffill().fillna(0.0)
    else:
        df["TNX_Change"] = 0.0

    # 5. Індекс напівпровідників та ШІ-чіпів: SOXX ETF
    if soxx_df is not None and not soxx_df.empty and "Close" in soxx_df:
        soxx_copy = soxx_df.copy()
        soxx_copy.index = pd.to_datetime(soxx_copy.index, utc=True).normalize()
        soxx_ret = soxx_copy["Close"].pct_change(fill_method=None)
        df["SOXX_Return"] = soxx_ret.reindex(df.index).ffill().fillna(df["SP500_Return"])
    else:
        df["SOXX_Return"] = df["SP500_Return"]

    # Базові доходності акції на різних часових масштабах
    df["Daily_Return"] = df["Close"].pct_change(fill_method=None)
    df["Return_5d"] = df["Close"].pct_change(periods=5, fill_method=None).fillna(df["Daily_Return"])
    df["Return_20d"] = df["Close"].pct_change(periods=20, fill_method=None).fillna(df["Return_5d"])

    # Відносна сила акції проти S&P 500 та проти сектору напівпровідників SOXX
    df["Relative_Return_1d"] = (df["Daily_Return"] - df["SP500_Return"]).fillna(0.0)
    df["Relative_to_SOXX"] = (df["Daily_Return"] - df["SOXX_Return"]).fillna(0.0)
    df["VIX_Change"] = df["VIX_Close"].pct_change(fill_method=None).fillna(0.0)

    # Відносні відстані до ковзних середніх (замість сирих цін MA_5, MA_20, Close)
    ma_5 = df["Close"].rolling(window=5, min_periods=1).mean()
    ma_20 = df["Close"].rolling(window=20, min_periods=1).mean()
    ma_50 = df["Close"].rolling(window=50, min_periods=1).mean()
    ma_200 = df["Close"].rolling(window=200, min_periods=1).mean()

    df["Distance_to_MA5"] = ((df["Close"] - ma_5) / ma_5.replace(0, 1e-9)).fillna(0.0)
    df["Distance_to_MA20"] = ((df["Close"] - ma_20) / ma_20.replace(0, 1e-9)).fillna(0.0)
    df["Distance_to_MA50"] = ((df["Close"] - ma_50) / ma_50.replace(0, 1e-9)).fillna(0.0)
    df["Distance_to_MA200"] = ((df["Close"] - ma_200) / ma_200.replace(0, 1e-9)).fillna(0.0)

    # Волатильність закриття (короткострокова та місячна)
    df["Volatility_5"] = df["Daily_Return"].rolling(window=5, min_periods=1).std().fillna(0.0)
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20, min_periods=1).std().fillna(df["Volatility_5"])

    # 6. Garman-Klass Historical Volatility (враховує High/Low/Open/Close всередині дня — у 7 разів чутливіша за std)
    log_hl = np.log((df["High"] / df["Low"].replace(0, 1e-9)).clip(lower=1.0))
    log_co = np.log((df["Close"] / df["Open"].replace(0, 1e-9)).clip(lower=1e-9))
    gk_var = (0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)).clip(lower=0.0)
    df["GK_Volatility"] = np.sqrt(gk_var.rolling(window=10, min_periods=1).mean()).fillna(df["Volatility_5"])

    # Внутрішньоденна структура сесії та нічні гепи
    df["Intraday_Return"] = ((df["Close"] - df["Open"]) / df["Open"].replace(0, 1e-9)).fillna(0.0)
    df["Day_Range"] = ((df["High"] - df["Low"]) / df["Low"].replace(0, 1e-9)).fillna(0.0)
    prev_close = df["Close"].shift(1)
    df["Gap"] = ((df["Open"] - prev_close) / prev_close.replace(0, 1e-9)).fillna(0.0)

    # Календарні та об'ємні фактори
    df["Day_of_Week"] = df.index.day_of_week
    vol_ma15 = df["Volume"].rolling(window=15, min_periods=1).mean()
    df["Volume_Ratio"] = (df["Volume"] / vol_ma15.replace(0, 1e-9)).fillna(1.0)

    # Сезон квартальних звітів + Точний календар звітів компанії
    df["Month"] = df.index.month
    df["Earnings_Season"] = df["Month"].isin([1, 4, 7, 10]).astype(int)
    df = _apply_earnings_features(df, earnings_df)

    # ATR (Average True Range) — шоковий імпульс та відносна амплітуда
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=1).mean()
    df["ATR_Ratio"] = (tr / atr_14.replace(0, 1e-9)).fillna(1.0)
    df["ATR_Pct"] = (atr_14 / df["Close"].replace(0, 1e-9)).fillna(0.0)

    # 7. Volatility Squeeze (Bollinger Band Width vs Keltner ATR Channel):
    # Коли < 1.0, смуги Боллінджера зайшли всередину каналу Кельтнера (пружина стиснулась перед вибухом ціни!)
    bb_std = df["Close"].rolling(window=20, min_periods=2).std().fillna(0.0)
    bb_width_pct = (4.0 * bb_std) / ma_20.replace(0, 1e-9)
    keltner_width_pct = (3.0 * atr_14) / ma_20.replace(0, 1e-9)
    df["Volatility_Squeeze"] = (bb_width_pct / keltner_width_pct.replace(0, 1e-9)).clip(0.0, 5.0).fillna(1.0)

    # Позиція ціни всередині смуг Боллінджера (Bollinger Band Z-Score)
    df["BB_Position"] = ((df["Close"] - ma_20) / (2.0 * bb_std.replace(0, 1e-9))).clip(-3.0, 3.0).fillna(0.0)

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

    return df


def calculate_ticker_metrics(df_eval, current_ticker):
    """
    Уніфікований розрахунок метрик точності та відносного зсуву (bias_pct) для тікера
    на основі останніх 14 календарних днів.
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
