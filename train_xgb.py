import os
import warnings
import joblib
import pandas as pd
import yfinance as yf
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from curl_cffi import requests
from huggingface_hub import HfApi

warnings.simplefilter(action='ignore', category=FutureWarning)


def send_telegram_report(text):
    """Надсилає звіт в Телеграм із розбиттям на безпечні чанки за рядками"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("ℹ️ TELEGRAM_BOT_TOKEN або TELEGRAM_CHAT_ID не знайдені. Пропускаємо сповіщення.")
        return

    MAX_LEN = 3800
    chunks = []
    current_chunk = ""

    for line in text.split("\n"):
        if len(current_chunk) + len(line) + 1 > MAX_LEN:
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
                payload.pop("parse_mode")
                res_plain = requests.post(url, json=payload, timeout=15)
                if res_plain.status_code == 200:
                    print(f"🚀 Частину {idx}/{len(chunks)} надіслано звичайним текстом (fallback)!")
                else:
                    print(f"❌ Помилка Telegram API (частина {idx}): {res.text}")
        except Exception as e:
            print(f"❌ Не вдалося зв'язатися з Telegram: {e}")


def calculate_ticker_metrics(df_eval, current_ticker):
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


def train_and_upload():
    target_tickers = os.getenv("STOCK_TICKER", "AAPL")
    tickers = list(dict.fromkeys(t.strip() for t in target_tickers.split(",") if t.strip()))

    hf_token = os.getenv("HF_TOKEN")
    hf_model_repo = os.getenv("HF_MODEL_REPO_XGB", "nadtoka/predictive-stock-models-xgb")
    hf_repo_inside = os.getenv("HF_REPO", "nadtoka/predictive-stock-dataset")

    os.makedirs("models_xgb", exist_ok=True)

    # Завантажуємо окрему історію аудиту XGBoost
    df_eval = pd.DataFrame()
    if hf_repo_inside:
        try:
            eval_url = f"https://huggingface.co/datasets/{hf_repo_inside}/raw/main/evaluation_history_xgb.csv"
            df_eval = pd.read_csv(eval_url)
            if not df_eval.empty and "actual_price" in df_eval.columns:
                df_eval = df_eval.dropna(subset=["actual_price", "mae_pct"])
                df_eval = df_eval[df_eval["actual_price"] > 0]
            print(f"📋 Завантажено базу оцінювання evaluation_history_xgb.csv ({len(df_eval)} рядків).")
        except Exception as e:
            print(f"ℹ️ Не вдалося завантажити evaluation_history_xgb.csv (працюємо без компенсації): {e}")

    tg_report = "⚡ ЕКСПЕРИМЕНТ XGBOOST (v3.1 Gradient Boosting)\n"
    tg_report += "━━━━━━━━━━━━━━━━\n\n"

    daily_predictions = []
    successful_models = 0

    for ticker in tickers:
        try:
            data_path = f"data/{ticker}_history.csv"
            if not os.path.exists(data_path):
                print(f"⚠️ Файл даних для {ticker} не знайдено. Пропускаємо.")
                continue

            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True).normalize()

            if len(df) < 35:
                print(f"⚠️ Для {ticker} недостатньо історії ({len(df)} рядків, потрібно >= 35). Пропускаємо.")
                continue

            print(f"⚡ Обробка та XGBoost тренування для {ticker}...")
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

            # Фічі з еластичними ковзними середніми для IPO
            df["MA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
            df["MA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
            df["Daily_Return"] = df["Close"].pct_change(fill_method=None)
            df["Volatility_5"] = df["Daily_Return"].rolling(window=5, min_periods=1).std().fillna(0)

            df["Intraday_Return"] = (df["Close"] - df["Open"]) / df["Open"]
            df["Day_Range"] = (df["High"] - df["Low"]) / df["Low"]
            df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

            df["Day_of_Week"] = df.index.dayofweek
            df["Volume_MA15"] = df["Volume"].rolling(window=15, min_periods=1).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_MA15"]
            df["MA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
            df["Distance_to_MA200"] = (df["Close"] - df["MA_200"]) / df["MA_200"]
            df["Month"] = df.index.month
            df["Earnings_Season"] = df["Month"].isin([1, 4, 7, 10]).astype(int)
            df["PE_Ratio"] = df["Close"] / eps
            df["PS_Ratio"] = df["Close"] / rev_per_share
            df["Revenue_Growth"] = rev_growth
            if target_mean_price is not None and target_mean_price > 0:
                df["Analyst_Upside"] = (target_mean_price - df["Close"]) / df["Close"]
            else:
                df["Analyst_Upside"] = 0.0

            if (
                forward_pe is not None
                and trailing_pe is not None
                and forward_pe > 0
                and trailing_pe > 0
            ):
                df["PE_Expansion"] = float(forward_pe / trailing_pe)
            else:
                df["PE_Expansion"] = 1.0

            if recommendation_mean is not None and 1.0 <= recommendation_mean <= 5.0:
                df["Analyst_Score"] = float(recommendation_mean)
            else:
                df["Analyst_Score"] = 2.5

            prev_close = df["Close"].shift(1)
            tr1 = df["High"] - df["Low"]
            tr2 = (df["High"] - prev_close).abs()
            tr3 = (df["Low"] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = tr.rolling(window=14, min_periods=1).mean()
            df["ATR_Ratio"] = (tr / atr_14.replace(0, 1e-9)).fillna(1.0)

            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-9)
            df["RSI_14"] = 100 - (100 / (1 + rs))

            feature_cols = [
                "Close", "Volume", "MA_5", "MA_20", "Daily_Return", "Volatility_5",
                "Intraday_Return", "Day_Range", "Gap", "Day_of_Week", "Volume_Ratio",
                "SP500_Return", "VIX_Close", "RSI_14", "Distance_to_MA200",
                "Earnings_Season", "PE_Ratio", "PS_Ratio", "Revenue_Growth", "Analyst_Upside", "PE_Expansion", "Analyst_Score", "ATR_Ratio",
            ]

            df = df.dropna(subset=feature_cols)
            if df.empty or len(df) < 25:
                print(f"❌ Недостатньо валідних рядків після створення ознак для {ticker}.")
                continue

            latest_features = df[feature_cols].tail(1).copy()
            current_price = latest_features["Close"].values[0]
            vix_current = latest_features["VIX_Close"].values[0]

            df["Target_1d"] = df["Close"].pct_change(fill_method=None).shift(-1)
            df["Target_5d"] = df["Close"].pct_change(periods=5, fill_method=None).shift(-5)
            df["Target_20d"] = df["Close"].pct_change(periods=20, fill_method=None).shift(-20)
            df = df.dropna(subset=["Target_1d", "Target_5d", "Target_20d"])

            if df.empty or len(df) < 20:
                print(f"❌ Недостатньо тарґетів для {ticker}.")
                continue

            test_size = max(3, min(20, int(len(df) * 0.15)))
            train_df = df.iloc[:-test_size]
            test_df = df.iloc[-test_size:]

            X_train, y_train = train_df[feature_cols], train_df[["Target_1d", "Target_5d", "Target_20d"]]
            X_test, y_test = test_df[feature_cols], test_df[["Target_1d", "Target_5d", "Target_20d"]]

            if len(X_train) < 5 or len(X_test) == 0:
                print(f"❌ Недостатньо тренувальних зразків для {ticker} (train: {len(X_train)}). Пропускаємо.")
                continue

            # XGBoost з контролем перенавчання
            base_xgb = XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            model = MultiOutputRegressor(base_xgb)
            model.fit(X_train, y_train)

            # Валідація
            predictions = model.predict(X_test)
            mae_1d, mae_5d, mae_20d = mean_absolute_error(y_test, predictions, multioutput="raw_values")
            mae_1d_pct = mae_1d * 100
            mae_5d_pct = mae_5d * 100
            mae_20d_pct = mae_20d * 100
            mae_1d_usd = current_price * mae_1d
            mae_5d_usd = current_price * mae_5d
            mae_20d_usd = current_price * mae_20d

            preds = model.predict(latest_features)[0]
            tomorrow_pred = current_price * (1 + preds[0])
            week_pred = current_price * (1 + preds[1])
            month_pred = current_price * (1 + preds[2])

            # Feedback loop
            ticker_metrics = calculate_ticker_metrics(df_eval, ticker)
            bias_1d = ticker_metrics["1d"]["bias"]
            bias_5d = ticker_metrics["5d"]["bias"]
            bias_20d = ticker_metrics["20d"]["bias"]
            count_1d = ticker_metrics["1d"]["count"]
            count_5d = ticker_metrics["5d"]["count"]
            count_20d = ticker_metrics["20d"]["count"]
            win_rate_1d = ticker_metrics["1d"]["win_rate"]

            if vix_current <= 22.0:
                if count_1d >= 5:
                    cap_1d = 0.005 * current_price
                    tomorrow_pred += max(-cap_1d, min(bias_1d * 0.2, cap_1d))
                if count_5d >= 5:
                    cap_5d = 0.015 * current_price
                    week_pred += max(-cap_5d, min(bias_5d * 0.2, cap_5d))
                if count_20d >= 5:
                    cap_20d = 0.030 * current_price
                    month_pred += max(-cap_20d, min(bias_20d * 0.2, cap_20d))
            else:
                month_pred = current_price * (1 + preds[2])

            if count_1d < 5:
                badge = "🟡"
            elif win_rate_1d >= 65.0:
                badge = "🟢"
            elif win_rate_1d >= 45.0:
                badge = "🟡"
            else:
                badge = "🔴"

            # Збереження локально
            model_path = f"models_xgb/{ticker}_model.joblib"
            joblib.dump(model, model_path)
            successful_models += 1

            last_date = latest_features.index[-1]
            ua_days = {0: "Понеділок", 1: "Вівторок", 2: "Середа", 3: "Четвер", 4: "П'ятниця", 5: "Субота", 6: "Неділя"}

            date_1d = last_date + pd.offsets.BusinessDay(1)
            date_5d = last_date + pd.offsets.BusinessDay(5)
            date_20d = last_date + pd.offsets.BusinessDay(20)

            day_1d_text = f"{ua_days[date_1d.weekday()]} ({date_1d.strftime('%d.%m')})"
            day_5d_text = f"{ua_days[date_5d.weekday()]} ({date_5d.strftime('%d.%m')})"
            day_20d_text = f"{ua_days[date_20d.weekday()]} ({date_20d.strftime('%d.%m')})"

            daily_predictions.append({
                "date": last_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "current_price": float(current_price),
                "pred_1d": float(tomorrow_pred),
                "pred_5d": float(week_pred),
                "pred_20d": float(month_pred),
            })

            emoji_1d = "📈" if tomorrow_pred > current_price else "📉"
            emoji_5d = "🚀" if week_pred > current_price else "📉"
            emoji_20d = "🚀" if month_pred > current_price else "📉"
            win_rate_str = f" [WR: {win_rate_1d:.0f}%]" if count_1d >= 5 else ""

            tg_report += f"🔹 {badge} *{ticker}* (Поточна: ${current_price:.2f}){win_rate_str}:\n"
            tg_report += f"  • {day_1d_text} {emoji_1d}: ${tomorrow_pred:.2f} | MAE: {mae_1d_pct:.2f}% (${mae_1d_usd:.2f})\n"
            tg_report += f"  • {day_5d_text} {emoji_5d}: ${week_pred:.2f} | MAE: {mae_5d_pct:.2f}% (${mae_5d_usd:.2f})\n"
            tg_report += f"  • {day_20d_text} (місяць) {emoji_20d}: ${month_pred:.2f} | MAE: {mae_20d_pct:.2f}% (${mae_20d_usd:.2f})\n\n"

        except Exception as e:
            print(f"❌ Помилка тренування XGBoost для {ticker}: {e}")

    # Пуш моделей в окремий репозиторій XGBoost
    if hf_token and hf_model_repo and successful_models > 0:
        try:
            print(f"📦 Пуш оновлених XGBoost моделей на Hugging Face Hub ({hf_model_repo})...")
            api = HfApi()
            api.upload_folder(
                folder_path="models_xgb",
                repo_id=hf_model_repo,
                repo_type="model",
                token=hf_token,
            )
            tg_report += f"☁️ *Hugging Face:* XGBoost моделі успішно синхронізовано."
        except Exception as e:
            tg_report += f"⚠️ *Hugging Face:* Помилка завантаження ваг: {e}"

    send_telegram_report(tg_report)

    # Збереження історії в окремий predictions_history_xgb.csv
    if hf_token and hf_repo_inside and daily_predictions:
        try:
            print("💾 Синхронізація predictions_history_xgb.csv з Hugging Face...")
            api = HfApi()
            file_path = "predictions_history_xgb.csv"
            remote_url = f"https://huggingface.co/datasets/{hf_repo_inside}/raw/main/{file_path}"

            try:
                df_history = pd.read_csv(remote_url)
            except Exception:
                df_history = pd.DataFrame(columns=["date", "ticker", "current_price", "pred_1d", "pred_5d", "pred_20d"])

            df_new = pd.DataFrame(daily_predictions)
            df_combined = pd.concat([df_history, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["date", "ticker"], keep="last")

            csv_data = df_combined.to_csv(index=False)
            api.upload_file(
                path_or_fileobj=csv_data.encode("utf-8"),
                path_in_repo=file_path,
                repo_id=hf_repo_inside,
                repo_type="dataset",
                token=hf_token,
            )
            print("✅ predictions_history_xgb.csv засинкронено в Hugging Face!")
        except Exception as e:
            print(f"⚠️ Помилка синхронізації predictions_history_xgb.csv: {e}")


if __name__ == "__main__":
    train_and_upload()