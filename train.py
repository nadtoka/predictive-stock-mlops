import os
import warnings
import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from curl_cffi import requests
from huggingface_hub import HfApi

# Глушимо FutureWarning від Pandas/yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)


def send_telegram_report(text):
    """Надсилає фінальний аналітичний звіт в Телеграм чат"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ℹ️ TELEGRAM_BOT_TOKEN або TELEGRAM_CHAT_ID не знайдені. Пропускаємо сповіщення.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("🚀 Аналітичний звіт успішно надіслано в Telegram!")
        else:
            print(f"❌ Помилка Telegram API: {res.text}")
    except Exception as e:
        print(f"❌ Не вдалося зв'язатися з Telegram: {e}")


def calculate_ticker_metrics(df_eval, current_ticker):
    """
    Розрахунок метрик для тікера на основі останніх 14 календарних днів.
    Якщо в вікні менше 5 свіжих оцінок — не підтягуємо хвіст, а повертаємо нульовий bias.
    """
    metrics = {
        "1d": {"win_rate": 50.0, "bias": 0.0, "count": 0},
        "5d": {"win_rate": 50.0, "bias": 0.0, "count": 0},
    }
    if df_eval is None or df_eval.empty or "ticker" not in df_eval.columns:
        return metrics

    df_ticker = df_eval[df_eval["ticker"] == current_ticker].copy()
    if df_ticker.empty or "target_date" not in df_ticker.columns:
        return metrics

    df_ticker["target_date"] = pd.to_datetime(df_ticker["target_date"], utc=True).dt.tz_localize(None)
    now_date = pd.Timestamp.now().tz_localize(None).floor("D")
    cutoff_date = now_date - pd.Timedelta(days=14)

    for horizon in ["1d", "5d"]:
        df_h = df_ticker[df_ticker["horizon"] == horizon].copy()
        if df_h.empty:
            continue

        df_h = df_h.sort_values("target_date")
        df_recent = df_h[df_h["target_date"] >= cutoff_date]

        # Мінімум для включення в динамічну компенсацію — 5 свіжих оцінок.
        # Не "доповнюємо" вибірку хвостом, якщо даних недостатньо: це знижує якість сигналу.
        if len(df_recent) < 5:
            metrics[horizon] = {
                "win_rate": 50.0,
                "bias": 0.0,
                "count": int(len(df_recent)),
            }
            continue

        count = len(df_recent)
        win_rate = 50.0
        if "direction_correct" in df_recent.columns and not df_recent["direction_correct"].dropna().empty:
            win_rate = float(df_recent["direction_correct"].mean() * 100)

        bias = 0.0
        if "actual_price" in df_recent.columns and "predicted_price" in df_recent.columns:
            diff = df_recent["actual_price"] - df_recent["predicted_price"]
            if not diff.dropna().empty:
                bias = float(diff.median())

        if pd.isna(win_rate):
            win_rate = 50.0
        if pd.isna(bias):
            bias = 0.0

        metrics[horizon] = {
            "win_rate": win_rate,
            "bias": bias,
            "count": count,
        }

    return metrics


def train_and_upload():
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = list(dict.fromkeys(t.strip() for t in target_tickers.split(",") if t.strip()))

    hf_token = os.getenv("HF_TOKEN")
    hf_model_repo = os.getenv("HF_MODEL_REPO")
    hf_repo_inside = os.getenv("HF_REPO")

    os.makedirs("models", exist_ok=True)

    # 1. 🚀 Завантаження бази оцінки з Hugging Face для Continuous Learning
    df_eval = pd.DataFrame()
    if hf_repo_inside:
        try:
            eval_url = f"https://huggingface.co/datasets/{hf_repo_inside}/raw/main/evaluation_history.csv"
            df_eval = pd.read_csv(eval_url)
            print(f"📋 Завантажено базу оцінювання evaluation_history.csv ({len(df_eval)} рядків).")
        except Exception as e:
            print(f"ℹ️ Не вдалося завантажити evaluation_history.csv (працюємо без компенсації): {e}")

    # Заголовок нашого щонічного звіту
    tg_report = "📊 КВАНТОВИЙ АНАЛІЗ РИНКУ (v3.1 Continuous Learning)\n"
    tg_report += "━━━━━━━━━━━━━━━━\n\n"

    daily_predictions = []
    successful_models = 0

    for ticker in tickers:
        data_path = f"data/{ticker}_history.csv"
        if not os.path.exists(data_path):
            print(f"⚠️ Файл даних для {ticker} не знайдено. Пропускаємо.")
            continue

        print(f"🧠 Обробка та тренування моделі для {ticker}...")
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

        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).normalize()

        # S&P 500
        sp500_path = "data/SP500_history.csv"
        if os.path.exists(sp500_path):
            sp500_df = pd.read_csv(sp500_path, index_col=0, parse_dates=True)
            sp500_df.index = pd.to_datetime(sp500_df.index, utc=True).normalize()
            df["SP500_Return"] = sp500_df["Close"].pct_change(fill_method=None)
        else:
            df["SP500_Return"] = 0

        # VIX
        vix_path = "data/VIX_history.csv"
        if os.path.exists(vix_path):
            vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)
            vix_df.index = pd.to_datetime(vix_df.index, utc=True).normalize()
            df["VIX_Close"] = vix_df["Close"]
        else:
            df["VIX_Close"] = 15.0

        # Індикатори (Фічі)
        df["MA_5"] = df["Close"].rolling(window=5).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["Daily_Return"] = df["Close"].pct_change(fill_method=None)
        df["Volatility_5"] = df["Daily_Return"].rolling(window=5).std()

        df["Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]
        df["Day_Range"] = (df["High"] - df["Low"]) / df["Low"]
        df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

        df["Day_of_Week"] = df.index.dayofweek
        df["Volume_MA15"] = df["Volume"].rolling(window=15).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA15"]
        df.loc[:, "MA_200"] = df["Close"].rolling(window=200).mean()
        df.loc[:, "Distance_to_MA200"] = (df["Close"] - df["MA_200"]) / df["MA_200"]
        df.loc[:, "Month"] = df.index.month
        df.loc[:, "Earnings_Season"] = df["Month"].isin([1, 4, 7, 10]).astype(int)
        df.loc[:, "PE_Ratio"] = df["Close"] / eps
        df.loc[:, "PS_Ratio"] = df["Close"] / rev_per_share
        df.loc[:, "Revenue_Growth"] = rev_growth

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        feature_cols = [
            "Close", "Volume", "MA_5", "MA_20", "Daily_Return", "Volatility_5",
            "Intraday_Return", "Day_Range", "Gap", "Day_of_Week", "Volume_Ratio",
            "SP500_Return", "VIX_Close", "RSI_14", "Distance_to_MA200",
            "Earnings_Season", "PE_Ratio", "PS_Ratio", "Revenue_Growth",
        ]

        df = df.dropna(subset=feature_cols)
        latest_features = df[feature_cols].tail(1).copy()
        current_price = latest_features["Close"].values[0]

        df["Target_1d"] = df["Close"].pct_change(fill_method=None).shift(-1)
        df["Target_5d"] = df["Close"].pct_change(periods=5, fill_method=None).shift(-5)
        df = df.dropna(subset=["Target_1d", "Target_5d"])

        if df.empty:
            print(f"❌ Недостатньо даних після створення індикаторів для {ticker}.")
            continue

        train_df = df.iloc[:-20]
        test_df = df.iloc[-20:]

        X_train, y_train = train_df[feature_cols], train_df[["Target_1d", "Target_5d"]]
        X_test, y_test = test_df[feature_cols], test_df[["Target_1d", "Target_5d"]]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Валідація
        predictions = model.predict(X_test)
        mae_1d, mae_5d = mean_absolute_error(y_test, predictions, multioutput="raw_values")
        mae_1d_pct = mae_1d * 100
        mae_5d_pct = mae_5d * 100
        mae_1d_usd = current_price * mae_1d
        mae_5d_usd = current_price * mae_5d

        # Базові прогнози ШІ
        preds = model.predict(latest_features)[0]
        tomorrow_pred = current_price * (1 + preds[0])
        week_pred = current_price * (1 + preds[1])

        # 2. 🧠 CONTINUOUS LEARNING: Коригування зсуву та визначення бейджа надійності
        ticker_metrics = calculate_ticker_metrics(df_eval, ticker)
        bias_1d = ticker_metrics["1d"]["bias"]
        bias_5d = ticker_metrics["5d"]["bias"]
        count_1d = ticker_metrics["1d"]["count"]
        count_5d = ticker_metrics["5d"]["count"]
        win_rate_1d = ticker_metrics["1d"]["win_rate"]

        damping_factor = 0.2
        vix_current = float(df["VIX_Close"].iloc[-1]) if "VIX_Close" in df.columns and not df["VIX_Close"].dropna().empty else 15.0

        capped_bias_1d = 0.0
        capped_bias_5d = 0.0

        # 1d: компенсація лише при >= 5 свіжих оцінок, з демпфінгом 20% і капом ±0.5%
        if count_1d >= 5:
            cap_limit_1d = 0.005 * current_price
            capped_bias_1d = max(-cap_limit_1d, min(bias_1d * damping_factor, cap_limit_1d))
            if vix_current > 22:
                capped_bias_1d = 0.0
            tomorrow_pred += capped_bias_1d

        # 5d: компенсація лише при >= 5 свіжих оцінок, з демпфінгом 20% і капом ±1.5%
        if count_5d >= 5:
            cap_limit_5d = 0.015 * current_price
            capped_bias_5d = max(-cap_limit_5d, min(bias_5d * damping_factor, cap_limit_5d))
            if vix_current > 22:
                capped_bias_5d = 0.0
            week_pred += capped_bias_5d

        # Визначаємо емодзі-бейдж надійності
        if count_1d < 5:
            badge = "🟡"
        elif win_rate_1d >= 65.0:
            badge = "🟢"
        elif win_rate_1d >= 45.0:
            badge = "🟡"
        else:
            badge = "🔴"

        # Зберігаємо ваги моделі
        model_path = f"models/{ticker}_model.joblib"
        joblib.dump(model, model_path)
        successful_models += 1

        last_date = latest_features.index[-1]
        ua_days = {0: "Понеділок", 1: "Вівторок", 2: "Середа", 3: "Четвер", 4: "П'ятниця", 5: "Субота", 6: "Неділя"}

        date_1d = last_date + pd.offsets.BusinessDay(1)
        date_5d = last_date + pd.offsets.BusinessDay(5)

        day_1d_text = f"{ua_days[date_1d.weekday()]} ({date_1d.strftime('%d.%m')})"
        day_5d_text = f"{ua_days[date_5d.weekday()]} ({date_5d.strftime('%d.%m')})"

        # 3. 💾 Зберігаємо у лог ВЖЕ скориговані прогнози
        daily_predictions.append(
            {
                "date": last_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "current_price": float(current_price),
                "pred_1d": float(tomorrow_pred),
                "pred_5d": float(week_pred),
            }
        )

        emoji_1d = "📈" if tomorrow_pred > current_price else "📉"
        emoji_5d = "🚀" if week_pred > current_price else "📉"

        win_rate_str = f" [WR: {win_rate_1d:.0f}%]" if count_1d >= 5 else ""

        # Формуємо рядок звіту
        tg_report += f"🔹 {badge} *{ticker}* (Поточна: ${current_price:.2f}){win_rate_str}:\n"
        tg_report += f"  • {day_1d_text} {emoji_1d}: ${tomorrow_pred:.2f} | MAE: {mae_1d_pct:.2f}% (${mae_1d_usd:.2f})\n"
        tg_report += f"  • {day_5d_text} {emoji_5d}: ${week_pred:.2f} | MAE: {mae_5d_pct:.2f}% (${mae_5d_usd:.2f})\n\n"

    # Синхронізація моделей на HF
    if hf_token and hf_model_repo and successful_models > 0:
        try:
            print("📦 Пуш оновлених моделей на Hugging Face Hub...")
            api = HfApi()
            api.upload_folder(
                folder_path="models",
                repo_id=hf_model_repo,
                repo_type="model",
                token=hf_token,
            )
            tg_report += "☁️ *Hugging Face:* Моделі успішно синхронізовано."
        except Exception as e:
            tg_report += f"⚠️ *Hugging Face:* Помилка завантаження ваг: {e}"
    else:
        tg_report += "ℹ️ *Hugging Face:* Синхронізацію пропущено (немає токенів)."

    send_telegram_report(tg_report)

    # Збереження історії прогнозів на HF
    if hf_token and hf_repo_inside and daily_predictions:
        try:
            print("💾 Синхронізація історії прогнозів з Hugging Face...")

            api = HfApi()
            file_path = "predictions_history.csv"
            remote_url = f"https://huggingface.co/datasets/{hf_repo_inside}/raw/main/{file_path}"

            try:
                df_history = pd.read_csv(remote_url)
                print("📜 Знайдено існуючу історію прогнозів. Оновлюємо...")
            except Exception:
                df_history = pd.DataFrame(columns=["date", "ticker", "current_price", "pred_1d", "pred_5d"])
                print("✨ Створюємо новий файл історії прогнозів...")

            df_new = pd.DataFrame(daily_predictions)
            df_combined = pd.concat([df_history, df_new], ignore_index=True)

            csv_data = df_combined.to_csv(index=False)
            api.upload_file(
                path_or_fileobj=csv_data.encode("utf-8"),
                path_in_repo=file_path,
                repo_id=hf_repo_inside,
                repo_type="dataset",
                token=hf_token,
            )
            print("✅ Історію прогнозів успішно засинкронено в Hugging Face!")
        except Exception as e:
            print(f"⚠️ Не вдалося зберегти історію прогнозів на HF (але пайплайн продовжує роботу): {e}")


if __name__ == "__main__":
    train_and_upload()